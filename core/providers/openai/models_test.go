package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

func TestListModelsByKeyResponseShapes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		body        string
		wantID      string
		wantOwnedBy string
		wantContext int
	}{
		{
			name:        "OpenAI envelope",
			body:        `{"object":"list","data":[{"id":"gpt-5","owned_by":"openai","context_window":128000}]}`,
			wantID:      "test/gpt-5",
			wantOwnedBy: "openai",
			wantContext: 128000,
		},
		{
			name:        "Together array",
			body:        `[{"id":"zai-org/GLM-5.2","organization":"Z.ai","context_length":131072}]`,
			wantID:      "test/zai-org/GLM-5.2",
			wantOwnedBy: "Z.ai",
			wantContext: 131072,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()

			response, bifrostErr := ListModelsByKey(
				schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
				&fasthttp.Client{},
				server.URL,
				schemas.Key{Models: schemas.WhiteList{"*"}},
				false,
				nil,
				schemas.ModelProvider("test"),
				false,
				false,
			)

			require.Nil(t, bifrostErr)
			require.Len(t, response.Data, 1)
			require.Equal(t, test.wantID, response.Data[0].ID)
			require.Equal(t, schemas.Ptr(test.wantOwnedBy), response.Data[0].OwnedBy)
			require.Equal(t, schemas.Ptr(test.wantContext), response.Data[0].ContextLength)
		})
	}
}

func TestModelRetrieveResponseShape(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"gpt-6-astra","object":"model","created":1686935002,"owned_by":"openai","shutdown_date":"2026-10-23"}`))
	}))
	defer server.Close()

	response, bifrostErr := HandleOpenAIModelRetrieveRequest(
		schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
		&fasthttp.Client{},
		server.URL+"/v1/models/gpt-6-astra",
		schemas.Key{},
		nil,
		schemas.ModelProvider("test"),
		false,
		false,
	)

	require.Nil(t, bifrostErr)
	require.Equal(t, "test/gpt-6-astra", response.ID)
	require.Equal(t, schemas.Ptr("openai"), response.OwnedBy)
	require.Equal(t, schemas.Ptr(int64(1686935002)), response.Created)
	require.Equal(t, schemas.Ptr("2026-10-23"), response.ShutdownDate)
}

func TestModelRetrieveUpstreamErrorIsPropagated(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"The model 'nope' does not exist","type":"invalid_request_error","code":"model_not_found"}}`))
	}))
	defer server.Close()

	response, bifrostErr := HandleOpenAIModelRetrieveRequest(
		schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
		&fasthttp.Client{},
		server.URL+"/v1/models/nope",
		schemas.Key{},
		nil,
		schemas.ModelProvider("test"),
		false,
		false,
	)

	require.Nil(t, response)
	require.NotNil(t, bifrostErr)
	require.Equal(t, schemas.Ptr(http.StatusNotFound), bifrostErr.StatusCode)
	require.Contains(t, bifrostErr.Error.Message, "does not exist")
}

func TestListModelsExtendedMetadataAndStrataShape(t *testing.T) {
	t.Parallel()

	// Covers anti-api (modalities + capabilities.vision), llama-swap (architecture + capabilities.vision),
	// and strata (status object + context_length/max_model_len + architecture).
	const body = `{"object":"list","data":[` +
		`{"id":"gemini-4-hydrogen","owned_by":"antigravity","display_name":"Gemini 4 Hydrogen","context_length":1048576,"max_input_tokens":1048576,"max_output_tokens":262144,"capabilities":{"vision":true},"modalities":{"input":["text","image","audio","video"],"output":["text"]}},` +
		`{"id":"lfm3-vl-2b","owned_by":"llama-swap","name":"lfm3-vl-2b","context_length":65536,"capabilities":{"vision":true},"architecture":{"input_modalities":["text","image","video"],"output_modalities":["text"]},"status":{"value":"loaded"}},` +
		`{"id":"inkling-2-mini-ud-q4_k_xl","object":"model","status":{"value":"loaded"},"max_model_len":524288,"context_length":524288,"max_context_length":524288,"meta":{"n_ctx":524288},"architecture":{"input_modalities":["text","image"],"output_modalities":["text"]}},` +
		`{"id":"cap-only-vision","capabilities":{"vision":true}}` +
		`]}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	response, bifrostErr := ListModelsByKey(
		schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
		&fasthttp.Client{}, server.URL,
		schemas.Key{Models: schemas.WhiteList{"*"}},
		false, nil,
		schemas.ModelProvider("test"),
		false, false,
	)

	require.Nil(t, bifrostErr)
	require.Len(t, response.Data, 4)

	gem := response.Data[0]
	require.Equal(t, "test/gemini-4-hydrogen", gem.ID)
	require.Equal(t, schemas.Ptr("Gemini 4 Hydrogen"), gem.Name)
	require.Equal(t, schemas.Ptr(1048576), gem.ContextLength)
	require.Equal(t, schemas.Ptr(1048576), gem.MaxInputTokens)
	require.Equal(t, schemas.Ptr(262144), gem.MaxOutputTokens)
	require.Equal(t, &schemas.Architecture{
		InputModalities:  []string{"text", "image", "audio", "video"},
		OutputModalities: []string{"text"},
	}, gem.Architecture)

	lfm := response.Data[1]
	require.Equal(t, "test/lfm3-vl-2b", lfm.ID)
	require.Equal(t, schemas.Ptr(65536), lfm.ContextLength)
	require.Equal(t, []string{"text", "image", "video"}, lfm.Architecture.InputModalities)

	strata := response.Data[2]
	require.Equal(t, "test/inkling-2-mini-ud-q4_k_xl", strata.ID)
	require.Equal(t, schemas.Ptr(524288), strata.ContextLength)
	require.Equal(t, []string{"text", "image"}, strata.Architecture.InputModalities)

	capOnly := response.Data[3]
	require.Equal(t, []string{"text", "image"}, capOnly.Architecture.InputModalities)
}
