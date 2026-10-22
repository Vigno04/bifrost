package handlers

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/fasthttp/router"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/warp"
	"github.com/valyala/fasthttp"
)

// loadedWarpManagement loads the repo spec into a management MCP whose tool
// calls are served by root, and returns it as Warp's management tools.
func loadedWarpManagement(t *testing.T, root fasthttp.RequestHandler, allowWrites bool) *warpManagementTools {
	t.Helper()
	SetLogger(&mockLogger{})
	spec, err := filepath.Abs(repoOpenAPISpec)
	if err != nil {
		t.Fatal(err)
	}
	exec := NewManagementMCPExecutor(root, 4<<20)
	t.Cleanup(func() { _ = exec.Close() })
	m := NewManagementMCP(ManagementMCPOptions{SpecSources: []string{spec}, AllowWrites: allowWrites, ExecClient: exec.Client})
	if err := m.Load(context.Background()); err != nil {
		t.Fatalf("loading spec: %v", err)
	}
	return NewWarpManagementTools(m).(*warpManagementTools)
}

// Warp is offered every read except the log reads its own tools answer, and the
// creates; never an update, a delete or a destructive POST. Creates appear only
// when the management MCP serves writes at all.
func TestWarpManagementCatalogOffersReadsAndCreates(t *testing.T) {
	catalog := loadedWarpManagement(t, func(*fasthttp.RequestCtx) {}, true).Catalog()
	if catalog.Len() == 0 {
		t.Fatal("empty catalog")
	}
	tools := map[string]bool{}
	for _, name := range []string{"listVirtualKeys", "getConfig", "createVirtualKey", "getLogs", "getMCPLogs", "updateVirtualKey", "deleteVirtualKey", "rotateVirtualKey"} {
		tools[name] = warpCatalogHas(catalog, name)
	}
	want := map[string]bool{
		"listVirtualKeys": true, "getConfig": true, "createVirtualKey": true,
		"getLogs": false, "getMCPLogs": false,
		"updateVirtualKey": false, "deleteVirtualKey": false, "rotateVirtualKey": false,
	}
	for name, offered := range want {
		if tools[name] != offered {
			t.Errorf("%s offered=%v, want %v", name, tools[name], offered)
		}
	}

	readOnly := loadedWarpManagement(t, func(*fasthttp.RequestCtx) {}, false).Catalog()
	if warpCatalogHas(readOnly, "createVirtualKey") || !warpCatalogHas(readOnly, "listVirtualKeys") {
		t.Error("with writes off Warp must get reads only")
	}
}

// warpCatalogHas reports whether a catalog offers a tool, and as a write when
// the name says it creates.
func warpCatalogHas(catalog *warp.ManagementCatalog, name string) bool {
	tool, ok := catalog.Tool(name)
	return ok && tool.Write == strings.HasPrefix(name, "create")
}

// A management call from Warp reaches /api as the dashboard user: the session
// the auth middleware verified from the cookie becomes a Bearer credential, and
// the cookie itself is never forwarded.
func TestWarpManagementCallRunsAsTheDashboardUser(t *testing.T) {
	var seenAuth, seenCookie string
	r := router.New()
	r.GET("/api/providers", func(ctx *fasthttp.RequestCtx) {
		seenAuth = string(ctx.Request.Header.Peek("Authorization"))
		seenCookie = string(ctx.Request.Header.Peek("Cookie"))
		ctx.SetContentType("application/json")
		ctx.SetBodyString(`{"providers":[],"total":0}`)
	})
	tools := loadedWarpManagement(t, r.Handler, false)

	var req fasthttp.Request
	req.Header.SetCookie("token", "session-abc")
	dashboard := initCtx(&req)
	dashboard.SetUserValue(schemas.BifrostContextKeySessionToken, "session-abc")

	// Through the snapshot the chat route takes, which is what outlives the request.
	snapshot, cancel, err := snapshotWarpContext(dashboard, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	result, failed, err := tools.Call(snapshot, "listProviders", map[string]any{})
	if err != nil || failed {
		t.Fatalf("call failed: %v %q", err, result)
	}
	if seenAuth != "Bearer session-abc" {
		t.Errorf("Authorization %q, want the session as a Bearer credential", seenAuth)
	}
	if seenCookie != "" {
		t.Errorf("cookie forwarded: %q", seenCookie)
	}

	// A credential the request already carries is forwarded as it is.
	var withHeader fasthttp.Request
	withHeader.Header.Set("Authorization", "Basic YWRtaW46cHc=")
	dashboard = initCtx(&withHeader)
	dashboard.SetUserValue(schemas.BifrostContextKeySessionToken, "session-abc")
	if _, _, err := tools.Call(warpManagementCaller(context.Background(), dashboard), "listProviders", map[string]any{}); err != nil {
		t.Fatal(err)
	}
	if seenAuth != "Basic YWRtaW46cHc=" {
		t.Errorf("Authorization %q, want the request's own", seenAuth)
	}

	// No caller on ctx means no credential, which /api would refuse.
	if _, _, err := tools.Call(context.Background(), "listProviders", map[string]any{}); err != nil {
		t.Fatal(err)
	}
	if seenAuth != "" {
		t.Errorf("Authorization %q sent without a caller", seenAuth)
	}
}

// BenchmarkWarpManagementCaller is what each Warp chat request now spends
// recording its caller for management calls.
func BenchmarkWarpManagementCaller(b *testing.B) {
	var req fasthttp.Request
	req.Header.SetCookie("token", "session-abc")
	req.Header.Set("User-Agent", "Mozilla/5.0")
	dashboard := initCtx(&req)
	dashboard.SetUserValue(schemas.BifrostContextKeySessionToken, "session-abc")
	b.ReportAllocs()
	for b.Loop() {
		_ = warpManagementCaller(context.Background(), dashboard)
	}
}
