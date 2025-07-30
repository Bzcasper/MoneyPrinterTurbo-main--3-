# Gemini API OpenAPI Specification

---

## Overview

This document describes the OpenAPI specification for the Gemini generative language endpoint.

- **Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`
- **Authentication:** API key via `Authorization` header

---

## Required Headers

| Header         | Value                | Description                |
|---------------|----------------------|----------------------------|
| Authorization | Bearer YOUR_API_KEY  | API key for authentication |
| Content-Type  | application/json     | Request payload format     |

---

## Request Payload Schema

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        { "text": "Your prompt here" }
      ]
    }
  ]
}
```

- **contents**: Array of message objects
- **role**: "user" or "model"
- **parts**: Array of content parts (text, images, etc.)

---

## Response Schema

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          { "text": "Generated response text" }
        ]
      }
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 12,
    "candidateTokenCount": 24,
    "totalTokenCount": 36
  }
}
```

- **candidates**: Array of generated responses
- **usageMetadata**: Token usage details

---

## Example Request

```http
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "contents": [
    {
      "role": "user",
      "parts": [
        { "text": "Hello, Gemini!" }
      ]
    }
  ]
}
```

---

## Example Response

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          { "text": "Hello! How can I assist you today?" }
        ]
      }
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 5,
    "candidateTokenCount": 9,
    "totalTokenCount": 14
  }
}
```

---

## Notes

- Refer to [Google Gemini API docs](https://ai.google.dev/docs/gemini_api_overview) for full details.
- Do not expose your API key in public documentation.
  