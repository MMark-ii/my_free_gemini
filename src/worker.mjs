import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error("Error caught by errHandler:",err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      console.log(`[DEBUG] Received request for URL: ${request.url}, Pathname for routing: ${pathname}`);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
        case pathname.endsWith("/images/generations"):
          assert(request.method === "POST");
          return handleImageGeneration(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/images/edits"):
          assert(request.method === "POST");
          return handleImageEditing(await request.json(), apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

const API_CLIENT = "genai-js/0.15.0"; // Updated to match @google/generative-ai version in package.json
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels (apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
const DEFAULT_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"; // ИСПОЛЬЗУЕМ МОДЕЛЬ ИЗ ДОКУМЕНТАЦИИ

async function handleEmbeddings (req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    if (!req.model.startsWith("gemini-")) {
      req.model = DEFAULT_EMBEDDINGS_MODEL;
    }
    model = "models/" + req.model;
  }
  if (!Array.isArray(req.input)) {
    req.input = [ req.input ];
  }
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-1.5-flash"; // Изменено на 1.5-flash, так как gemini-2.0-flash не существует

// Helper function for generating a mock chat completion ID
const generateId = (length = 29) => {
  const PUSH_CHARS = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz';
  let lastPushTime = 0;
  let lastRandChars = [];
  let now = new Date().getTime();
  let duplicateTime = (now === lastPushTime);
  lastPushTime = now;
  let timeStampChars = new Array(8);
  for (let i = 7; i >= 0; i--) {
    timeStampChars[i] = PUSH_CHARS.charAt(now % 64);
    now = Math.floor(now / 64);
  }
  if (now !== 0) throw new Error('We should have converted the entire timestamp.');
  let id = timeStampChars.join('');
  if (!duplicateTime) {
    for (let i = 0; i < 12; i++) {
      lastRandChars[i] = Math.floor(Math.random() * 64);
    }
  } else {
    let i;
    for (i = 11; i >= 0 && lastRandChars[i] === 63; i--) {
      lastRandChars[i] = 0;
    }
    lastRandChars[i]++;
  }
  for (let i = 0; i < 12; i++) {
    id += PUSH_CHARS.charAt(lastRandChars[i]);
  }
  return id.slice(0, length);
};


// Placeholder for transformRequest, processCompletionsResponse, parseStream, etc.
// These would need to be defined as in the original openai-gemini project for chat completions to fully work.
// For simplicity in this example, we'll assume they exist or are not critical for image generation focus.
// If chat completions are also needed, these functions must be copied from the original project.

async function transformRequest(req) {
  const {
    messages,
    model,
    temperature,
    top_p,
    max_tokens,
    stop,
    tools,
    tool_choice,
    // поля, которые НЕ должны попасть в тело Gemini
    stream,
    stream_options,
    parallel_tool_calls,
    ...rest // прочие поля, которые не обрабатываем – лучше не передавать
  } = req;

  // 1. Преобразуем сообщения в содержимое contents
  const systemMessages = messages.filter(m => m.role === 'system');
  const otherMessages = messages.filter(m => m.role !== 'system');

  // Системная инструкция (только одна, последняя)
  let system_instruction = undefined;
  if (systemMessages.length > 0) {
    const lastSystem = systemMessages[systemMessages.length - 1];
    if (typeof lastSystem.content === 'string') {
      system_instruction = {
        parts: [{ text: lastSystem.content }]
      };
    }
    // Если content – массив, можно взять все текстовые части, но обычно system – строка
  }

  // Преобразуем остальные сообщения (user, assistant, tool)
  const contents = [];
  for (const msg of otherMessages) {
    const role = msg.role === 'assistant' ? 'model' : msg.role; // assistant -> model
    // role 'tool' в Gemini отсутствует, обычно результаты функций вставляются как functionResponse части
    if (role === 'tool') {
      // Обработка tool-сообщений (результаты вызова функций)
      // Здесь нужно будет преобразовать в part с functionResponse, но для простоты пока пропустим
      // или вызовем ошибку, т.к. базовая реализация может не поддерживать tools.
      // Пока просто пропустим такие сообщения, чтобы не сломать запрос.
      continue;
    }

    let parts = [];
    if (typeof msg.content === 'string') {
      parts.push({ text: msg.content });
    } else if (Array.isArray(msg.content)) {
      // Преобразуем OpenAI-части в Gemini parts
      for (const part of msg.content) {
        if (part.type === 'text') {
          parts.push({ text: part.text });
        } else if (part.type === 'image_url') {
          // Извлекаем base64-данные из data URL
          const url = part.image_url?.url;
          if (url && url.startsWith('data:')) {
            const [header, data] = url.split(',');
            const mimeType = header.split(':')[1].split(';')[0];
            parts.push({
              inlineData: {
                mimeType,
                data
              }
            });
          }
          // URL-изображения напрямую не поддерживаются в Gemini inlineData,
          // потребуется предварительная загрузка и преобразование (пока опустим)
        }
        // другие типы (image, audio) можно добавить аналогично при необходимости
      }
    }
    contents.push({ role, parts });
  }

  // 2. Собираем generationConfig
  const generationConfig = {};
  if (temperature !== undefined) generationConfig.temperature = temperature;
  if (top_p !== undefined) generationConfig.topP = top_p;
  if (max_tokens !== undefined) {
    generationConfig.maxOutputTokens = max_tokens;
  }
  if (stop !== undefined) {
    // stop может быть строкой или массивом строк
    generationConfig.stopSequences = Array.isArray(stop) ? stop : [stop];
  }

  // 3. Преобразуем tools и tool_choice
  let toolsGemini = undefined;
  let toolConfig = undefined;

  if (tools && Array.isArray(tools) && tools.length > 0) {
    // Gemini ожидает массив объектов Tool, каждый из которых содержит functionDeclarations
    // Обычно все функции кладут в один Tool
    const functionDeclarations = [];
    for (const tool of tools) {
      if (tool.type === 'function' && tool.function) {
        const func = tool.function;
        // Параметры должны быть в формате OpenAPI JSON Schema (parameters)
        // Убедимся, что есть name, description (необязательно) и parameters
        functionDeclarations.push({
          name: func.name,
          description: func.description || '',
          parameters: func.parameters, // JSON Schema объекта
        });
      }
    }
    if (functionDeclarations.length > 0) {
      toolsGemini = [{ functionDeclarations }];
    }
  }

  if (tool_choice) {
    // Преобразуем OpenAI tool_choice в Gemini ToolConfig
    // tool_choice может быть "auto", "none", "required" (ANY), или объект { type: "function", function: { name: "..." } }
    let mode = 'AUTO';
    let allowedFunctionNames = undefined;
    if (typeof tool_choice === 'string') {
      if (tool_choice === 'none') mode = 'NONE';
      else if (tool_choice === 'auto') mode = 'AUTO';
      else if (tool_choice === 'required') mode = 'ANY';
    } else if (tool_choice && tool_choice.type === 'function') {
      mode = 'ANY';
      allowedFunctionNames = [tool_choice.function.name];
    }
    toolConfig = {
      functionCallingConfig: {
        mode,
        ...(allowedFunctionNames && { allowedFunctionNames })
      }
    };
  }

  // 4. Формируем финальный объект для Gemini
  const requestBody = {
    contents,
    ...(system_instruction && { systemInstruction: system_instruction }),
    ...(Object.keys(generationConfig).length > 0 && { generationConfig }),
    ...(toolsGemini && { tools: toolsGemini }),
    ...(toolConfig && { toolConfig })
  };

  // Примечание: поля stream, stream_options, parallel_tool_calls намеренно исключены
  return requestBody;
}

function processCompletionsResponse(body, model, id) {
  // Basic response processing
  const choice = {
    index: 0,
    message: {
      role: "assistant",
      content: body.candidates[0].content.parts[0].text,
    },
    finish_reason: body.candidates[0].finishReason || "stop", // Adjust based on actual Gemini reasons
  };
  return {
    id,
    choices: [choice],
    created: Math.floor(Date.now() / 1000),
    model,
    object: "chat.completion",
    // usage data would need to be calculated or estimated
  };
}


async function handleCompletions (req, apiKey) {
  let modelName = DEFAULT_MODEL;
  if (typeof req.model === "string") {
    if (req.model.startsWith("models/")) {
      modelName = req.model.substring(7);
    } else if (req.model.startsWith("gemini-") || req.model.startsWith("gemma-")) {
      modelName = req.model;
    }
  }
  
  let transformedBody = await transformRequest(req);
  
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${modelName}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(transformedBody),
  });

  let responseBody = response.body;
  if (response.ok) {
    if (req.stream) {
      // Streaming logic would go here, taken from original project.
      // For now, let's return an error for streaming as it's complex.
      // This part needs to be implemented properly if streaming is required.
      console.warn("Streaming for chat completions is not fully implemented in this version.");
      throw new HttpError("Streaming not fully implemented", 501);

    } else {
      const jsonBody = await response.json();
      if (!jsonBody.candidates || !jsonBody.candidates[0].content) {
         console.error("Invalid completion object from Gemini:", jsonBody);
         throw new Error("Invalid completion object from Gemini");
      }
      const id = "chatcmpl-" + generateId();
      responseBody = processCompletionsResponse(jsonBody, modelName, id);
      return new Response(JSON.stringify(responseBody), fixCors(response));
    }
  } else {
     const errorText = await response.text();
     console.error(`Gemini API Error for completions: ${errorText}`);
     throw new HttpError(`Error from Gemini API: ${response.statusText} - ${errorText}`, response.status);
  }
   // Fallback for stream, though ideally handled above
  return new Response(responseBody, fixCors(response));
}


const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  // "HARM_CATEGORY_CIVIC_INTEGRITY", // Usually kept commented out or handled carefully
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE", // Or a more restrictive setting
}));


async function handleImageEditing(req, apiKey) {
  // req здесь - это уже распарсенное тело запроса (json), 
  // так как в switch мы передаем await request.json()
  const { prompt, n = 1, image, mime_type } = req; 

  if (!prompt) {
    // Используем HttpError для консистентности с другими обработчиками
    throw new HttpError("Prompt is required for editing", 400);
  }
  if (!image || !mime_type) {
    throw new HttpError("Input image (base64) and mime_type are required for editing", 400);
  }

  const modelForGeminiApi = DEFAULT_IMAGE_MODEL; 
  // safetySettings должны быть определены глобально или переданы

  const parts = [
    { text: prompt }, 
    {
      inlineData: { 
        mimeType: mime_type,
        data: image,
      },
    },
  ];

  const geminiPayload = {
    contents: [{ parts: parts }],
    generationConfig: {
      candidateCount: n,
      response_modalities: ['TEXT', 'IMAGE'], 
    },
    safetySettings, 
  };

  const apiUrl = `${BASE_URL}/${API_VERSION}/models/${modelForGeminiApi}:generateContent`;

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: makeHeaders(apiKey, { "Content-Type": "application/json" }), 
      body: JSON.stringify(geminiPayload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error("Error from Gemini API (handleImageEditing):", response.status, errorBody);
      throw new HttpError(`Error from Gemini API: ${errorBody}`, response.status);
    }

    const geminiResponse = await response.json();
    const imageData = [];

    if (geminiResponse.candidates && geminiResponse.candidates.length > 0) {
      for (const candidate of geminiResponse.candidates) {
        if (candidate.content && candidate.content.parts) {
          for (const part of candidate.content.parts) {
            if (part.inlineData && part.inlineData.mimeType && part.inlineData.mimeType.startsWith("image/")) {
              imageData.push({ b64_json: part.inlineData.data });
            }
            if (part.text) {
              console.log("Gemini (handleImageEditing) accompanying text:", part.text);
            }
          }
        }
      }
    }

    if (imageData.length === 0) {
      console.warn("No image data found in Gemini (handleImageEditing) response for prompt:", prompt, "Response:", JSON.stringify(geminiResponse, null, 2));
      if (geminiResponse.promptFeedback?.blockReason) {
        console.warn("Safety Ratings:", JSON.stringify(geminiResponse.promptFeedback.safetyRatings, null, 2));
        throw new HttpError(`Prompt blocked by Gemini for reason: ${geminiResponse.promptFeedback.blockReason}. Check safety ratings.`, 400);
      }
      throw new HttpError("No edited image data generated by Gemini, or image data not found in the expected format.", 500);
    }
    
    const openAiResponse = {
      created: Math.floor(Date.now() / 1000),
      data: imageData.slice(0, n), 
    };
    return new Response(JSON.stringify(openAiResponse), fixCors({ headers: { "Content-Type": "application/json" } }));

  } catch (error) {
    // Если это уже HttpError, просто перебрасываем его, чтобы errHandler его поймал
    if (error instanceof HttpError) throw error;
    // Иначе это непредвиденная ошибка
    console.error("Unexpected error in handleImageEditing:", error);
    throw new HttpError("Internal server error during image editing.", 500);
  }
}

async function handleImageGeneration(req, apiKey) {
  if (!apiKey) {
    throw new HttpError("Missing API key", 401);
  }

  const { prompt, n: requestedN = 1 } = req; 

  if (!prompt || typeof prompt !== 'string' || prompt.trim() === "") {
    throw new HttpError("Missing or invalid 'prompt' in request body", 400);
  }

  const n = Math.max(1, Math.min(parseInt(requestedN, 10) || 1, 4)); 

  // Структура запроса для Gemini API (для :generateContent)
  const geminiPayload = {
    contents: [{
      parts: [
        { text: prompt }
      ]
    }],
    generationConfig: {
      candidateCount: n,
      response_modalities: ['TEXT', 'IMAGE']
    },
    safetySettings, // Глобально определенные настройки безопасности
  };

  const modelForGeminiApi = DEFAULT_IMAGE_MODEL; 
  
  // Используем эндпоинт :generateContent для Gemini моделей
  const apiUrl = `${BASE_URL}/${API_VERSION}/models/${modelForGeminiApi}:generateContent`; 

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(geminiPayload),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    // Используем разные console.error для ясности
    console.error("Gemini API Error (generateContent for Image):", errorBody, "Status:", response.status, "URL:", apiUrl);
    throw new HttpError(`Error from Gemini API (generateContent for Image): ${response.statusText} - ${errorBody}`, response.status);
  }

  const geminiResponse = await response.json();

  const imageData = [];
  // Обработка ответа Gemini для :generateContent
  if (geminiResponse.candidates && geminiResponse.candidates.length > 0) {
    for (const candidate of geminiResponse.candidates) {
      if (candidate.content && candidate.content.parts) {
        for (const part of candidate.content.parts) {
          if (part.inlineData && part.inlineData.mimeType && part.inlineData.mimeType.startsWith("image/")) {
            imageData.push({ b64_json: part.inlineData.data });
          }
          if (part.text) {
            console.log("Gemini (generateContent for Image) accompanying text:", part.text);
          }
        }
      }
    }
  }

  if (imageData.length === 0) {
    console.warn("No image data found in Gemini (generateContent for Image) response for prompt:", prompt, "Response:", JSON.stringify(geminiResponse, null, 2));
    if (geminiResponse.promptFeedback?.blockReason) {
        if (geminiResponse.promptFeedback.safetyRatings) {
            console.warn("Safety Ratings:", JSON.stringify(geminiResponse.promptFeedback.safetyRatings, null, 2));
        }
        throw new HttpError(`Prompt blocked by Gemini (generateContent for Image) for reason: ${geminiResponse.promptFeedback.blockReason}. Check safety ratings in Vercel logs.`, 400);
    }
    throw new HttpError("No image data generated by Gemini (generateContent for Image), or image data not found in the expected format.", 500);
  }

  const openAiResponse = {
    created: Math.floor(Date.now() / 1000),
    data: imageData.slice(0, n), 
  };

  return new Response(JSON.stringify(openAiResponse), fixCors({ headers: { "Content-Type": "application/json" } }));
}
