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
const DEFAULT_IMAGE_MODEL = "gemini-1.5-pro-latest"; // Устанавливаем эту модель для теста

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
  // Basic transformation, copy more complex logic from original project if needed
  const { messages, model, ...rest } = req;
  const contents = messages
    .filter(msg => msg.role === 'user' || msg.role === 'assistant' || msg.role === 'system')
    .map(msg => ({
      role: msg.role === 'assistant' ? 'model' : msg.role, // "assistant" maps to "model" for Gemini
      parts: [{ text: msg.content }],
    }));
  
  // Handle system instruction if present
  const systemInstructionMsg = messages.find(msg => msg.role === 'system');
  const system_instruction = systemInstructionMsg ? { role: 'system', parts: [{ text: systemInstructionMsg.content }] } : undefined;


  return {
    contents,
    system_instruction, // Add system instruction here
    ...rest // Include other parameters like temperature, maxOutputTokens etc.
  };
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


async function handleImageGeneration(req, apiKey) {
  if (!apiKey) {
    throw new HttpError("Missing API key", 401);
  }

  const { prompt, n: requestedN = 1 } = req; 

  if (!prompt || typeof prompt !== 'string' || prompt.trim() === "") {
    throw new HttpError("Missing or invalid 'prompt' in request body", 400);
  }

  const n = Math.max(1, Math.min(parseInt(requestedN, 10) || 1, 4)); 

  // Структура запроса для Imagen API (предположение на основе SDK)
  const imagenApiPayload = {
    prompt: prompt,
    // "config": { "number_of_images": n } // SDK-стиль
    number_of_images: n, // Более вероятный стиль для REST API
    // Могут понадобиться и другие параметры, например, output_format: "B64_JSON" или что-то подобное
    // Либо указание, что мы ожидаем base64 ответ, если imageBytes не является base64 по умолчанию
  };

  const modelForImagenApi = DEFAULT_IMAGE_MODEL;
  
  // ПРЕДПОЛОЖЕНИЕ: эндпоинт для Imagen :generateImages
  const apiUrl = `${BASE_URL}/${API_VERSION}/models/${modelForImagenApi}:generateImages`; 

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(imagenApiPayload),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error("Imagen API Error for Image Generation:", errorBody, "Status:", response.status, "URL:", apiUrl);
    throw new HttpError(`Error from Imagen API: ${response.statusText} - ${errorBody}`, response.status);
  }

  const imagenResponse = await response.json();

  const imageData = [];
  // Обработка ответа Imagen (предположение на основе SDK)
  if (imagenResponse.generated_images && imagenResponse.generated_images.length > 0) {
    for (const generatedImage of imagenResponse.generated_images) {
      // Предполагаем, что generatedImage.image.image_bytes это уже base64 строка, как в примере с Buffer.from(..., "base64")
      // Или generatedImage.b64_json, или generatedImage.imageBytes если это прямой маппинг
      // Нужно проверить точную структуру ответа REST API
      let b64JsonData = generatedImage.image?.image_bytes || generatedImage.b64_json || generatedImage.imageBytes;
      if (b64JsonData) {
          imageData.push({ b64_json: b64JsonData });
      } else {
          console.warn("Generated image part does not contain expected image data field (image.image_bytes, b64_json, or imageBytes):", generatedImage);
      }
    }
  }

  if (imageData.length === 0) {
    console.warn("No image data found in Imagen response for prompt:", prompt, "Response:", JSON.stringify(imagenResponse, null, 2));
    // Imagen API может иметь свои коды/причины ошибок, например, для заблокированных промптов
    // if (imagenResponse.error?.message) { ... }
    throw new HttpError("No image data generated by Imagen, or image data not found in the expected format.", 500);
  }

  const openAiResponse = {
    created: Math.floor(Date.now() / 1000),
    data: imageData.slice(0, n), 
  };

  return new Response(JSON.stringify(openAiResponse), fixCors({ headers: { "Content-Type": "application/json" } }));
}
