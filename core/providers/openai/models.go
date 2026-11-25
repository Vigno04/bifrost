package openai

import (
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

func resolveOpenAIModelName(model *OpenAIModel) *string {
	if model == nil {
		return nil
	}
	if model.DisplayName != nil && *model.DisplayName != "" {
		return model.DisplayName
	}
	if model.Name != nil && *model.Name != "" {
		return model.Name
	}
	return nil
}

func resolveOpenAIModelArchitecture(model *OpenAIModel) *schemas.Architecture {
	if model == nil {
		return nil
	}
	var arch *schemas.Architecture
	if model.Architecture != nil {
		copyArch := *model.Architecture
		if model.Architecture.InputModalities != nil {
			copyArch.InputModalities = append([]string(nil), model.Architecture.InputModalities...)
		}
		if model.Architecture.OutputModalities != nil {
			copyArch.OutputModalities = append([]string(nil), model.Architecture.OutputModalities...)
		}
		arch = &copyArch
	} else if model.Modalities != nil {
		arch = &schemas.Architecture{
			InputModalities:  append([]string(nil), model.Modalities.Input...),
			OutputModalities: append([]string(nil), model.Modalities.Output...),
		}
	}

	if arch == nil || len(arch.InputModalities) == 0 {
		var visionFlag *bool
		if model.Capabilities != nil && model.Capabilities.Vision != nil {
			visionFlag = model.Capabilities.Vision
		} else if model.SupportsVision != nil {
			visionFlag = model.SupportsVision
		}
		if visionFlag != nil {
			if arch == nil {
				arch = &schemas.Architecture{OutputModalities: []string{"text"}}
			}
			if *visionFlag {
				arch.InputModalities = []string{"text", "image"}
			} else {
				arch.InputModalities = []string{"text"}
			}
		}
	}
	return arch
}

// ToBifrostListModelsResponse converts an OpenAI list models response to a Bifrost list models response
func (response *OpenAIListModelsResponse) ToBifrostListModelsResponse(providerKey schemas.ModelProvider, allowedModels schemas.WhiteList, blacklistedModels schemas.BlackList, aliases schemas.KeyAliases, unfiltered bool) *schemas.BifrostListModelsResponse {
	if response == nil {
		return nil
	}

	bifrostResponse := &schemas.BifrostListModelsResponse{
		Data: make([]schemas.Model, 0, len(response.Data)),
	}

	pipeline := &providerUtils.ListModelsPipeline{
		AllowedModels:     allowedModels,
		BlacklistedModels: blacklistedModels,
		Aliases:           aliases,
		Unfiltered:        unfiltered,
		ProviderKey:       providerKey,
		MatchFns:          providerUtils.DefaultMatchFns(),
	}
	if pipeline.ShouldEarlyExit() {
		return bifrostResponse
	}

	included := make(map[string]bool)

	for _, model := range response.Data {
		reconcileOpenAIContextWindow(&model)
		for _, result := range pipeline.FilterModel(model.ID) {
			entry := schemas.Model{
				ID:              string(providerKey) + "/" + result.ResolvedID,
				Name:            resolveOpenAIModelName(&model),
				Created:         model.Created,
				OwnedBy:         schemas.Ptr(model.OwnedBy),
				ContextLength:   model.ContextWindow,
				MaxInputTokens:  model.MaxInputTokens,
				MaxOutputTokens: model.MaxOutputTokens,
				Architecture:    resolveOpenAIModelArchitecture(&model),
				Reasoning:       toSchemasModelReasoning(model.Reasoning),
			}
			if result.AliasValue != "" {
				entry.Alias = schemas.Ptr(result.AliasValue)
			}
			bifrostResponse.Data = append(bifrostResponse.Data, entry)
			included[strings.ToLower(result.ResolvedID)] = true
		}
	}

	bifrostResponse.Data = append(bifrostResponse.Data,
		pipeline.BackfillModels(included)...)

	return bifrostResponse
}

// ToBifrostModelRetrieveResponse converts an OpenAI model object to a Bifrost model retrieve response
func (model *OpenAIModel) ToBifrostModelRetrieveResponse(providerKey schemas.ModelProvider) *schemas.BifrostModelRetrieveResponse {
	if model == nil {
		return nil
	}
	copyModel := *model
	reconcileOpenAIContextWindow(&copyModel)

	return &schemas.BifrostModelRetrieveResponse{
		Model: schemas.Model{
			ID:              string(providerKey) + "/" + copyModel.ID,
			Name:            resolveOpenAIModelName(&copyModel),
			Created:         copyModel.Created,
			OwnedBy:         schemas.Ptr(copyModel.OwnedBy),
			ContextLength:   copyModel.ContextWindow,
			MaxInputTokens:  copyModel.MaxInputTokens,
			MaxOutputTokens: copyModel.MaxOutputTokens,
			Architecture:    resolveOpenAIModelArchitecture(&copyModel),
			Reasoning:       toSchemasModelReasoning(copyModel.Reasoning),
			ShutdownDate:    copyModel.ShutdownDate,
		},
	}
}

// ToOpenAIModelRetrieveResponse converts a Bifrost model retrieve response to an OpenAI model object.
// The ID keeps its provider prefix so this surface agrees with ToOpenAIListModelsResponse.
func ToOpenAIModelRetrieveResponse(response *schemas.BifrostModelRetrieveResponse) *OpenAIModel {
	if response == nil {
		return nil
	}

	openaiModel := &OpenAIModel{
		ID:              response.ID,
		Object:          "model",
		Created:         response.Created,
		ShutdownDate:    response.ShutdownDate,
		MaxInputTokens:  response.MaxInputTokens,
		MaxOutputTokens: response.MaxOutputTokens,
		Name:            response.Name,
		Reasoning:       fromSchemasModelReasoning(response.Reasoning),
		Architecture:    response.Architecture,
	}
	if response.OwnedBy != nil {
		openaiModel.OwnedBy = *response.OwnedBy
	}
	openaiModel.ContextWindow = response.ContextLength
	if openaiModel.ContextWindow == nil {
		openaiModel.ContextWindow = response.MaxInputTokens
	}
	openaiModel.ContextLength = response.ContextLength
	if response.Architecture != nil && (len(response.Architecture.InputModalities) > 0 || len(response.Architecture.OutputModalities) > 0) {
		openaiModel.Modalities = &OpenAIModelModalities{
			Input:  response.Architecture.InputModalities,
			Output: response.Architecture.OutputModalities,
		}
	}
	return openaiModel
}

// ToOpenAIListModelsResponse converts a Bifrost list models response to an OpenAI list models response
func ToOpenAIListModelsResponse(response *schemas.BifrostListModelsResponse) *OpenAIListModelsResponse {
	if response == nil {
		return nil
	}
	openaiResponse := &OpenAIListModelsResponse{
		Data: make([]OpenAIModel, 0, len(response.Data)),
	}
	for _, model := range response.Data {
		openaiModel := OpenAIModel{
			ID:              model.ID,
			Object:          "model",
			MaxInputTokens:  model.MaxInputTokens,
			MaxOutputTokens: model.MaxOutputTokens,
			Name:            model.Name,
			Reasoning:       fromSchemasModelReasoning(model.Reasoning),
			Architecture:    model.Architecture,
		}
		if model.Created != nil {
			openaiModel.Created = model.Created
		}
		if model.OwnedBy != nil {
			openaiModel.OwnedBy = *model.OwnedBy
		}
		openaiModel.ContextWindow = model.ContextLength
		if openaiModel.ContextWindow == nil {
			openaiModel.ContextWindow = model.MaxInputTokens
		}
		openaiModel.ContextLength = model.ContextLength
		if model.Architecture != nil && (len(model.Architecture.InputModalities) > 0 || len(model.Architecture.OutputModalities) > 0) {
			openaiModel.Modalities = &OpenAIModelModalities{
				Input:  model.Architecture.InputModalities,
				Output: model.Architecture.OutputModalities,
			}
		}

		openaiResponse.Data = append(openaiResponse.Data, openaiModel)

	}
	return openaiResponse
}
