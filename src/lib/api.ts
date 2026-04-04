export interface DetectionResult {
  classifier: {
    label: "rubbish" | "notrubbish";
    confidence: number;
  };
  objects: Array<{
    label: string;
    confidence: number;
    bbox: [number, number, number, number]; // x1, y1, x2, y2 (relative 0-1)
    polygon?: [number, number][];            // mask contour points (relative 0-1)
    color: string;
  }>;
  annotated_image?: string;                 // base64 JPEG with YOLO masks drawn
}

export interface UploadedFile {
  id: string;
  file: File;
  url: string;
  result?: DetectionResult;
  processing?: boolean;
  error?: string;
}

export async function runInference(file: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Backend error ${response.status}: ${err}`);
  }

  const data = await response.json();
  return data as DetectionResult;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch("/api/health", { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
