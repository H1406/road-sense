import { useState, useCallback, useRef, useEffect } from "react";
import { CircleStop, Loader2, AlertTriangle, CheckCircle } from "lucide-react";
import { DetectionResult, runInference } from "../lib/api";

const CameraPage = () => {
  const [cameraActive, setCameraActive] = useState(false);
  const [liveResult, setLiveResult] = useState<DetectionResult | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopCamera = useCallback(() => {
    if (detectIntervalRef.current) clearInterval(detectIntervalRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
    setLiveResult(null);
    setIsDetecting(false);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(console.error);
        }
        setCameraActive(true);
        setError(null);

        detectIntervalRef.current = setInterval(async () => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (!video || !canvas || video.readyState < 2) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(video, 0, 0);
          canvas.toBlob(async (blob) => {
            if (!blob) return;
            setIsDetecting(true);
            try {
              const result = await runInference(new File([blob], "frame.jpg", { type: "image/jpeg" }));
              if (!cancelled) setLiveResult(result);
            } catch { /* silent */ } finally {
              setIsDetecting(false);
            }
          }, "image/jpeg", 0.85);
        }, 1200);
      } catch (err: any) {
        if (!cancelled) setError(err?.message ?? "Camera error");
      }
    };

    init();
    return () => {
      cancelled = true;
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="h-screen pt-16 flex flex-col overflow-hidden bg-background">
      <div className="flex-1 relative overflow-hidden" style={{ background: "hsl(var(--muted))" }}>
        <canvas ref={canvasRef} className="hidden" />

        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          playsInline
          autoPlay
          muted
        />

        {liveResult?.annotated_image && liveResult.classifier.label === "rubbish" && (
          <img
            src={liveResult.annotated_image}
            alt="detection overlay"
            className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            style={{ opacity: 0.85 }}
          />
        )}

        {!cameraActive && (
          <div className="absolute inset-0 flex items-center justify-center z-20 bg-background/70">
            {error ? (
              <div className="text-center">
                <p className="text-sm font-semibold text-destructive mb-2">Camera error</p>
                <p className="text-xs text-muted-foreground">{error}</p>
              </div>
            ) : (
              <div className="text-center">
                <Loader2 className="w-8 h-8 text-primary animate-spin mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">Starting camera...</p>
              </div>
            )}
          </div>
        )}

        <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
          <div className="flex items-center gap-2 glass rounded-lg px-3 py-1.5">
            <span className={`w-2 h-2 rounded-full ${cameraActive ? "bg-teal-400 animate-pulse" : "bg-muted-foreground"}`} />
            <span className="text-xs font-medium text-teal-400">{cameraActive ? "LIVE" : "..."}</span>
            {isDetecting && <Loader2 className="w-3 h-3 text-muted-foreground animate-spin ml-1" />}
          </div>
          {liveResult && (
            <div className={`flex items-center gap-1.5 glass rounded-lg px-3 py-1.5 ${liveResult.classifier.label === "rubbish" ? "text-destructive" : "text-teal-400"}`}>
              {liveResult.classifier.label === "rubbish" ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle className="w-3 h-3" />}
              <span className="text-xs font-medium">
                {liveResult.classifier.label === "rubbish" ? `Rubbish  ${liveResult.objects.length} obj` : "Clean"}
              </span>
              <span className="text-[10px] opacity-60">{(liveResult.classifier.confidence * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>

        {cameraActive && (
          <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10">
            <button
              onClick={stopCamera}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-destructive text-destructive-foreground text-sm font-semibold hover:brightness-110 transition-all shadow-lg"
            >
              <CircleStop className="w-4 h-4" /> Stop
            </button>
          </div>
        )}

        {liveResult && liveResult.objects.length > 0 && (
          <div className="absolute bottom-5 right-4 z-10 space-y-1 max-h-48 overflow-y-auto">
            {Object.values(
              liveResult.objects.reduce((acc, obj) => {
                if (!acc[obj.label]) acc[obj.label] = { ...obj, count: 0 };
                acc[obj.label].count += 1;
                if (obj.confidence > acc[obj.label].confidence) acc[obj.label].confidence = obj.confidence;
                return acc;
              }, {} as Record<string, typeof liveResult.objects[0] & { count: number }>)
            ).map((grp) => (
              <div key={grp.label} className="flex items-center gap-1.5 glass rounded-lg px-2.5 py-1.5 text-xs">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: grp.color }} />
                <span className="font-medium">{grp.label.replace(/_/g, " ")}</span>
                {grp.count > 1 && <span className="text-muted-foreground">x{grp.count}</span>}
                <span className="text-muted-foreground ml-auto pl-2">{(grp.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraPage;
