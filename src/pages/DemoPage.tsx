import { useState, useCallback, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Upload, Camera, X, Loader2, Eye, EyeOff, AlertTriangle, CheckCircle, Circle, CircleStop } from "lucide-react";
import { UploadedFile, DetectionResult, runInference } from "../lib/api";

const sampleImages = [
  "https://i.ibb.co/1YDqzGCV/pexels-photo-13839236.jpg",
  "https://i.ibb.co/PZS5ptVF/rac-cong-kenh-h3-4366-9592-jpg.webp",
  "https://i.ibb.co/vCDgjTMK/z564285377889120c2f2df7e9b0b3f642ea029d0f79735-17213512395261825587256.jpg",
  "https://i.ibb.co/sJzPcmnn/11824216-10153561098978487-2038938996-n.jpg",
];

const categories = [
  "Mattress", "Leftover_tire", "Furniture", "Abandoned_shopping_cart",
  "Trash", "Toy", "Garbage_bag", "Cardboard_box",
  "Appliance", "Wooden_crate", "Trash_pile", "Metal_scrap",
];

const DemoPage = () => {
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [cameraActive, setCameraActive] = useState(false);
  const [liveResult, setLiveResult] = useState<DetectionResult | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const dragCounter = useRef(0);
  const [hoveredLabel, setHoveredLabel] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [displayArea, setDisplayArea] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

  const computeDisplayArea = useCallback(() => {
    const c = containerRef.current;
    const img = imgRef.current;
    if (!c || !img || !img.naturalWidth || !img.naturalHeight) return;
    const cW = c.clientWidth, cH = c.clientHeight;
    const scale = Math.min(cW / img.naturalWidth, cH / img.naturalHeight);
    const dW = img.naturalWidth * scale, dH = img.naturalHeight * scale;
    setDisplayArea({ left: (cW - dW) / 2, top: (cH - dH) / 2, width: dW, height: dH });
  }, []);

  useEffect(() => {
    window.addEventListener("resize", computeDisplayArea);
    return () => window.removeEventListener("resize", computeDisplayArea);
  }, [computeDisplayArea]);

  useEffect(() => { setDisplayArea(null); }, [activeIndex]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setLiveResult(null);

      // Detection loop — send frame every 1.2s
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
            setLiveResult(result);
          } catch {
            // silent
          } finally {
            setIsDetecting(false);
          }
        }, "image/jpeg", 0.85);
      }, 1200);
    } catch (err) {
      console.error("Camera error:", err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (detectIntervalRef.current) clearInterval(detectIntervalRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
    setLiveResult(null);
    setIsDetecting(false);
  }, []);

  // Stop camera on unmount or when switching away
  useEffect(() => {
    if (mode !== "camera") stopCamera();
  }, [mode, stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const processFile = useCallback(async (file: File) => {
    const id = Math.random().toString(36).slice(2);
    const url = URL.createObjectURL(file);
    const newFile: UploadedFile = { id, file, url, processing: true };

    setFiles((prev) => {
      const next = [...prev, newFile];
      setActiveIndex(next.length - 1);
      return next;
    });

    try {
      const result = await runInference(file);
      setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, processing: false, result } : f)));
    } catch (err) {
      console.error("Inference failed:", err);
      setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, processing: false } : f)));
    }
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current += 1;
    if (dragCounter.current === 1) setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) setDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setDragOver(false);
      Array.from(e.dataTransfer.files).forEach(processFile);
    },
    [processFile]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) Array.from(e.target.files).forEach(processFile);
    },
    [processFile]
  );

  const loadSample = useCallback(async (url: string) => {
    const res = await fetch(url);
    const blob = await res.blob();
    processFile(new File([blob], "sample.jpg", { type: "image/jpeg" }));
  }, [processFile]);

  const removeFile = (id: string) => {
    setFiles((prev) => {
      const next = prev.filter((f) => f.id !== id);
      if (activeIndex >= next.length) setActiveIndex(Math.max(0, next.length - 1));
      return next;
    });
  };

  const active = files[activeIndex];

  return (
    <div className="h-screen pt-16 flex flex-col overflow-hidden bg-background">
      {/* Toolbar */}
      <div
        className="flex-shrink-0 flex items-center justify-between px-5 border-b border-border/40"
        style={{ height: 44 }}
      >
        <div className="flex items-center gap-3">
          {mode === "upload" && files.length > 0 && (
            <p className="text-xs text-muted-foreground">
              {files.length} image{files.length !== 1 ? "s" : ""}
            </p>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {/* Mode toggle */}
          <div className="flex items-center gap-0.5 p-0.5 rounded-lg bg-secondary/50 border border-border/30">
            <button
              onClick={() => setMode("upload")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === "upload"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Upload className="w-3 h-3" />
              Upload
            </button>
            <button
              onClick={() => { setMode("camera"); setCameraActive(false); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                mode === "camera"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Camera className="w-3 h-3" />
              Camera
            </button>
          </div>
        </div>
      </div>

      {/* Main workspace */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* ── Camera mode ── */}
        {mode === "camera" && (
          <div className="flex-1 flex overflow-hidden min-h-0 relative" style={{ background: "hsl(var(--muted))" }}>
            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} className="hidden" />

            {!cameraActive ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-20 h-20 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-5">
                    <Camera className="w-8 h-8 text-primary" />
                  </div>
                  <p className="text-sm font-semibold mb-1">Live Camera</p>
                  <p className="text-xs text-muted-foreground mb-6">Real-time rubbish detection</p>
                  <button
                    onClick={startCamera}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:brightness-110 transition-all"
                  >
                    <Circle className="w-3.5 h-3.5" /> Start Camera
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Video feed */}
                <video
                  ref={videoRef}
                  className="absolute inset-0 w-full h-full object-contain"
                  playsInline
                  muted
                />

                {/* Annotated image overlay from backend */}
                {liveResult?.annotated_image && liveResult.classifier.label === "rubbish" && (
                  <img
                    src={liveResult.annotated_image}
                    alt="detection overlay"
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                    style={{ opacity: 0.85 }}
                  />
                )}

                {/* Status bar */}
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
                  <div className="flex items-center gap-2 glass rounded-lg px-3 py-1.5">
                    <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
                    <span className="text-xs font-medium text-teal-400">LIVE</span>
                    {isDetecting && <Loader2 className="w-3 h-3 text-muted-foreground animate-spin ml-1" />}
                  </div>
                  {liveResult && (
                    <div
                      className={`flex items-center gap-1.5 glass rounded-lg px-3 py-1.5 ${
                        liveResult.classifier.label === "rubbish" ? "text-destructive" : "text-teal-400"
                      }`}
                    >
                      {liveResult.classifier.label === "rubbish"
                        ? <AlertTriangle className="w-3 h-3" />
                        : <CheckCircle className="w-3 h-3" />
                      }
                      <span className="text-xs font-medium">
                        {liveResult.classifier.label === "rubbish"
                          ? `Rubbish · ${liveResult.objects.length} obj`
                          : "Clean"}
                      </span>
                      <span className="text-[10px] opacity-60">
                        {(liveResult.classifier.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>

                {/* Stop button */}
                <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10">
                  <button
                    onClick={stopCamera}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-destructive text-destructive-foreground text-sm font-semibold hover:brightness-110 transition-all shadow-lg"
                  >
                    <CircleStop className="w-4 h-4" /> Stop
                  </button>
                </div>

                {/* Object list — bottom-right */}
                {liveResult && liveResult.objects.length > 0 && (
                  <div className="absolute bottom-5 right-4 z-10 space-y-1 max-h-48 overflow-y-auto">
                    {Object.values(
                      liveResult.objects.reduce(
                        (acc, obj) => {
                          if (!acc[obj.label]) acc[obj.label] = { ...obj, count: 0 };
                          acc[obj.label].count += 1;
                          if (obj.confidence > acc[obj.label].confidence) acc[obj.label].confidence = obj.confidence;
                          return acc;
                        },
                        {} as Record<string, typeof liveResult.objects[0] & { count: number }>
                      )
                    ).map((grp) => (
                      <div
                        key={grp.label}
                        className="flex items-center gap-1.5 glass rounded-lg px-2.5 py-1.5 text-xs"
                      >
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: grp.color }} />
                        <span className="font-medium">{grp.label.replace(/_/g, " ")}</span>
                        {grp.count > 1 && <span className="text-muted-foreground">×{grp.count}</span>}
                        <span className="text-muted-foreground ml-auto pl-2">{(grp.confidence * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ── Upload mode ── */}
        {mode === "upload" && (
          <>
        {/* Left column: upload + thumbnails (only when files exist) */}
        {files.length > 0 && (
          <div
            className={`w-64 xl:w-72 flex-shrink-0 flex flex-col border-r border-border/40 overflow-hidden transition-colors duration-150 ${
              dragOver ? "bg-primary/5" : ""
            }`}
            style={{ background: dragOver ? undefined : "hsl(var(--card))" }}
            onDragEnter={handleDragEnter}
            onDragOver={(e) => e.preventDefault()}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {/* Drop / upload button */}
            <label
              className={`flex-shrink-0 flex flex-col items-center justify-center gap-2 px-4 py-6 border-b cursor-pointer transition-all ${
                dragOver
                  ? "border-primary/40 text-primary bg-primary/5"
                  : "border-border/30 text-muted-foreground/50 hover:text-muted-foreground hover:bg-secondary/30"
              }`}
            >
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${
                dragOver ? "bg-primary/20 border border-primary/40" : "bg-secondary/40 border border-border/30"
              }`}>
                <Upload className="w-4 h-4" />
              </div>
              <div className="text-center">
                <p className="text-xs font-medium leading-snug">
                  {dragOver ? "Drop to add" : "Drop or Upload"}
                </p>
                <p className="text-[10px] text-muted-foreground/40 mt-0.5">image / video</p>
              </div>
              <input
                type="file"
                accept="image/*,video/*"
                multiple
                className="hidden"
                onChange={handleFileInput}
              />
            </label>

            {/* Thumbnails — vertical scroll */}
            <div className="flex-1 overflow-y-auto flex flex-col gap-1.5 p-3">
              {files.map((f, i) => (
                <div key={f.id} className="relative flex-shrink-0 group">
                  <button
                    onClick={() => setActiveIndex(i)}
                    className={`w-full h-16 rounded-lg overflow-hidden border-2 transition-all focus:outline-none ${
                      i === activeIndex ? "border-primary" : "border-border/40 hover:border-border"
                    }`}
                  >
                    <img src={f.url} alt="" className="w-full h-full object-cover" />
                    {f.processing && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <Loader2 className="w-3 h-3 text-primary animate-spin" />
                      </div>
                    )}
                  </button>
                  {f.result && (
                    <div
                      className={`absolute bottom-0 left-0 right-0 h-0.5 rounded-b-lg ${
                        f.result.classifier.label === "rubbish" ? "bg-destructive" : "bg-teal-400"
                      }`}
                    />
                  )}
                  <button
                    onClick={() => removeFile(f.id)}
                    className="absolute top-1 right-1 w-5 h-5 rounded-full bg-background/80 border border-border flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-10"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Center: image canvas */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          {/* Canvas */}
          <div
            ref={containerRef}
            className="flex-1 relative overflow-hidden"
            style={{ background: "hsl(var(--muted))" }}
            onDragEnter={handleDragEnter}
            onDragOver={(e) => e.preventDefault()}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {files.length === 0 ? (
              /* Empty state */
              <label
                className={`absolute inset-0 flex items-center justify-center transition-colors duration-200 cursor-pointer ${
                  dragOver ? "bg-primary/5" : ""
                }`}
              >
                <input type="file" accept="image/*,video/*" multiple className="hidden" onChange={handleFileInput} />
                <div className={`text-center transition-transform duration-200 ${dragOver ? "scale-105" : ""}`}>
                  <div
                    className={`border-2 border-dashed rounded-2xl mx-6 p-14 max-w-3xl transition-all ${
                      dragOver ? "border-primary/60" : "border-border/30 hover:border-border/50"
                    }`}
                  >
                    <Upload
                      className={`w-7 h-7 mx-auto mb-4 transition-colors ${
                        dragOver ? "text-primary" : "text-muted-foreground/30"
                      }`}
                    />
                    <p className="text-sm font-medium text-foreground mb-6">
                      {dragOver ? "Drop to analyse" : "Drop image or click to upload"}
                    </p>
                    <div onClick={(e) => e.stopPropagation()}>
                      <p className="text-[10px] uppercase tracking-widest text-muted-foreground/40 mb-3">
                        Try a sample
                      </p>
                      <div className="grid grid-cols-4 gap-3">
                        {sampleImages.map((url, i) => (
                          <button
                            key={i}
                            onClick={() => loadSample(url)}
                            className="aspect-[4/3] rounded-xl overflow-hidden border border-border/20 hover:border-primary/40 transition-all group"
                          >
                            <img
                              src={url}
                              alt={`sample ${i + 1}`}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                            />
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </label>
            ) : active ? (
              /* Active viewer */
              <div className="absolute inset-0">
                <img
                  ref={imgRef}
                  src={active.url}
                  alt=""
                  className="w-full h-full object-contain"
                  onLoad={computeDisplayArea}
                />

                {/* Processing overlay */}
                {active.processing && (
                  <div className="absolute inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-16 h-16 rounded-2xl border border-primary/20 bg-primary/10 flex items-center justify-center mx-auto mb-5">
                        <Loader2 className="w-7 h-7 text-primary animate-spin" />
                      </div>
                      <p className="text-sm font-semibold text-foreground mb-1">Running pipeline</p>
                      <p className="text-xs text-muted-foreground">Classification → Segmentation</p>
                    </div>
                  </div>
                )}

                {/* Annotated image overlay */}
                {active.result &&
                  showOverlay &&
                  active.result.classifier.label === "rubbish" &&
                  active.result.annotated_image && (
                    <div className="absolute inset-0 pointer-events-none">
                      <img
                        src={active.result.annotated_image}
                        alt="segmentation overlay"
                        className="w-full h-full object-contain transition-opacity duration-200"
                        style={{ opacity: hoveredLabel !== null ? 0.25 : 0.85 }}
                      />
                    </div>
                  )}

                {/* SVG hover highlight — all instances of hovered label */}
                {active.result &&
                  showOverlay &&
                  active.result.classifier.label === "rubbish" &&
                  hoveredLabel !== null &&
                  displayArea &&
                  (() => {
                    const matches = active.result!.objects.filter((o) => o.label === hoveredLabel);
                    if (!matches.length) return null;
                    const { left, top, width, height } = displayArea;
                    return (
                      <svg
                        className="absolute pointer-events-none"
                        style={{ left, top, width, height }}
                        viewBox="0 0 1 1"
                        preserveAspectRatio="none"
                      >
                        <rect x="0" y="0" width="1" height="1" fill="rgba(0,0,0,0.5)" />
                        {matches.map((obj, mi) =>
                          obj.polygon && obj.polygon.length >= 3 ? (
                            <polygon
                              key={mi}
                              points={obj.polygon.map(([x, y]) => `${x},${y}`).join(" ")}
                              fill={`${obj.color}55`}
                              stroke={obj.color}
                              strokeWidth="0.006"
                              strokeLinejoin="round"
                            />
                          ) : (
                            <rect
                              key={mi}
                              x={obj.bbox[0]}
                              y={obj.bbox[1]}
                              width={obj.bbox[2] - obj.bbox[0]}
                              height={obj.bbox[3] - obj.bbox[1]}
                              fill={`${obj.color}40`}
                              stroke={obj.color}
                              strokeWidth="0.006"
                            />
                          )
                        )}
                      </svg>
                    );
                  })()}

                {/* Bbox fallback (no annotated image) */}
                {active.result &&
                  showOverlay &&
                  active.result.classifier.label === "rubbish" &&
                  !active.result.annotated_image && (
                    <div className="absolute inset-0 pointer-events-none">
                      {active.result.objects.map((obj, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.15 }}
                          className="absolute border-2 rounded-lg"
                          style={{
                            left: `${obj.bbox[0] * 100}%`,
                            top: `${obj.bbox[1] * 100}%`,
                            width: `${(obj.bbox[2] - obj.bbox[0]) * 100}%`,
                            height: `${(obj.bbox[3] - obj.bbox[1]) * 100}%`,
                            borderColor: obj.color,
                            backgroundColor: `${obj.color}15`,
                          }}
                        >
                          <span
                            className="absolute -top-5 left-0 text-[10px] font-semibold px-1.5 py-0.5 rounded"
                            style={{ backgroundColor: obj.color, color: "#000" }}
                          >
                            {obj.label.replace(/_/g, " ")} {(obj.confidence * 100).toFixed(0)}%
                          </span>
                        </motion.div>
                      ))}
                    </div>
                  )}

                {/* Drag-over indicator */}
                {dragOver && (
                  <div className="absolute inset-0 bg-primary/10 border-2 border-primary/50 flex items-center justify-center z-10">
                    <div className="text-center">
                      <Upload className="w-8 h-8 text-primary mx-auto mb-2" />
                      <p className="text-sm font-medium">Drop to add</p>
                    </div>
                  </div>
                )}

                {/* Result badge */}
                {active.result && !active.processing && (
                  <div className="absolute top-3 left-3 z-10">
                    <div
                      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold backdrop-blur-sm ${
                        active.result.classifier.label === "rubbish"
                          ? "bg-destructive/80 text-white"
                          : "bg-teal-500/80 text-white"
                      }`}
                    >
                      {active.result.classifier.label === "rubbish" ? (
                        <AlertTriangle className="w-3 h-3" />
                      ) : (
                        <CheckCircle className="w-3 h-3" />
                      )}
                      {active.result.classifier.label === "rubbish" ? "Rubbish" : "Clean"}
                      <span className="opacity-75">
                        · {(active.result.classifier.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>

        {/* Right sidebar */}
        <div
          className="w-64 xl:w-72 flex-shrink-0 border-l border-border/40 overflow-y-auto"
          style={{ background: "hsl(var(--card))" }}
        >
          {files.length === 0 ? (
            /* Info panel */
            <div className="p-5 space-y-6">
              <div>
                <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest mb-4">
                  Pipeline
                </p>
                <div className="space-y-4">
                  {[
                    { n: "01", label: "Classification", detail: "97.1% accuracy", sub: "Rubbish vs Clean" },
                    { n: "02", label: "Segmentation", detail: "mAP@0.5 — 0.380", sub: "12-class instance masks" },
                  ].map(({ n, label, detail, sub }) => (
                    <div key={n} className="flex gap-3">
                      <span className="font-mono text-[10px] text-muted-foreground/30 pt-0.5 w-5 flex-shrink-0">
                        {n}
                      </span>
                      <div>
                        <p className="text-xs font-semibold text-foreground">{label}</p>
                        <p className="text-[11px] text-primary/70 mt-0.5">{detail}</p>
                        <p className="text-[11px] text-muted-foreground">{sub}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="h-px bg-border/20" />

              <div>
                <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest mb-3">
                  Categories
                </p>
                <div className="flex flex-wrap gap-1">
                  {categories.map((cat) => (
                    <span
                      key={cat}
                      className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-secondary/40 text-secondary-foreground"
                    >
                      {cat.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>

              <div className="h-px bg-border/20" />

              <div>
                <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest mb-3">
                  Model
                </p>
                <p className="text-[11px] font-mono text-muted-foreground leading-relaxed">
                  YOLO11l-cls
                  <br />
                  YOLO11l-seg
                </p>
                <p className="text-[10px] text-muted-foreground/50 mt-2">Trained on 6,626 images</p>
              </div>
            </div>
          ) : active?.processing ? (
            /* Loading state */
            <div className="p-5">
              <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest mb-5">
                Analysis
              </p>
              <div className="space-y-4">
                {[
                  { label: "Stage 1 — Classification", isActive: true },
                  { label: "Stage 2 — Segmentation", isActive: false },
                ].map(({ label, isActive }, i) => (
                  <div key={label} className="flex items-center gap-3">
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
                        isActive
                          ? "bg-primary/20 border border-primary/40"
                          : "bg-secondary/40 border border-border/20"
                      }`}
                    >
                      {isActive ? (
                        <Loader2 className="w-3 h-3 text-primary animate-spin" />
                      ) : (
                        <span className="text-[9px] font-bold text-muted-foreground/30">{i + 1}</span>
                      )}
                    </div>
                    <span className={`text-xs ${isActive ? "text-foreground" : "text-muted-foreground/40"}`}>
                      {label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : active?.result ? (
            /* Results */
            <div className="p-5 space-y-5">
              <div>
                <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest mb-3">
                  Classification
                </p>
                <div
                  className={`rounded-xl p-3 ${
                    active.result.classifier.label === "rubbish"
                      ? "bg-destructive/10 border border-destructive/20"
                      : "bg-teal-500/5 border border-teal-500/20"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    {active.result.classifier.label === "rubbish" ? (
                      <AlertTriangle className="w-3.5 h-3.5 text-destructive flex-shrink-0" />
                    ) : (
                      <CheckCircle className="w-3.5 h-3.5 text-teal-400 flex-shrink-0" />
                    )}
                    <div>
                      <p className="text-xs font-semibold">
                        {active.result.classifier.label === "rubbish" ? "Rubbish Detected" : "Clean Scene"}
                      </p>
                      <p className="text-[11px] text-muted-foreground">
                        {(active.result.classifier.confidence * 100).toFixed(1)}% confidence
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {active.result.classifier.label === "rubbish" && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest">
                        Segmentation
                      </p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        {active.result.objects.length} object
                        {active.result.objects.length !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <button
                      onClick={() => setShowOverlay(!showOverlay)}
                      className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-all"
                    >
                      {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                    </button>
                  </div>

                  {(() => {
                    // Generic/catch-all labels that tend to dominate
                    const GENERIC = new Set(["Trash", "Trash_pile"]);

                    // Group objects by label
                    const groupMap = active.result.objects.reduce(
                      (acc, obj) => {
                        if (!acc[obj.label]) {
                          acc[obj.label] = { label: obj.label, color: obj.color, count: 0, maxConf: 0 };
                        }
                        acc[obj.label].count += 1;
                        if (obj.confidence > acc[obj.label].maxConf) acc[obj.label].maxConf = obj.confidence;
                        return acc;
                      },
                      {} as Record<string, { label: string; color: string; count: number; maxConf: number }>
                    );

                    // Specific classes first (sorted by maxConf desc), generic ones at bottom
                    const groups = Object.values(groupMap).sort((a, b) => {
                      const aGeneric = GENERIC.has(a.label);
                      const bGeneric = GENERIC.has(b.label);
                      if (aGeneric !== bGeneric) return aGeneric ? 1 : -1;
                      return b.maxConf - a.maxConf;
                    });

                    const specificGroups = groups.filter((g) => !GENERIC.has(g.label));
                    const genericGroups = groups.filter((g) => GENERIC.has(g.label));

                    const renderRow = (grp: typeof groups[0], i: number, isGeneric: boolean) => (
                      <motion.button
                        key={grp.label}
                        initial={{ opacity: 0, x: 6 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        onMouseEnter={() => setHoveredLabel(grp.label)}
                        onMouseLeave={() => setHoveredLabel(null)}
                        className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all duration-100 ${
                          hoveredLabel === grp.label
                            ? "bg-primary/10 border border-primary/25"
                            : "hover:bg-secondary/40 border border-transparent"
                        }`}
                      >
                        <span
                          className="w-2 h-2 rounded-full flex-shrink-0"
                          style={{
                            backgroundColor: grp.color,
                            boxShadow: hoveredLabel === grp.label ? `0 0 6px ${grp.color}88` : "none",
                            opacity: isGeneric ? 0.5 : 1,
                          }}
                        />
                        <span className={`text-xs truncate flex-1 ${isGeneric ? "text-muted-foreground" : ""}`}>
                          {grp.label.replace(/_/g, " ")}
                        </span>
                        {grp.count > 1 && (
                          <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-secondary text-muted-foreground flex-shrink-0">
                            ×{grp.count}
                          </span>
                        )}
                        <span className="text-[10px] font-mono text-muted-foreground flex-shrink-0">
                          {(grp.maxConf * 100).toFixed(0)}%
                        </span>
                      </motion.button>
                    );

                    return (
                      <>
                        <div className="space-y-0.5">
                          {specificGroups.map((grp, i) => renderRow(grp, i, false))}
                          {genericGroups.length > 0 && specificGroups.length > 0 && (
                            <div className="flex items-center gap-2 py-1.5 px-1">
                              <div className="h-px flex-1 bg-border/20" />
                              <span className="text-[9px] text-muted-foreground/30 uppercase tracking-widest">general</span>
                              <div className="h-px flex-1 bg-border/20" />
                            </div>
                          )}
                          {genericGroups.map((grp, i) => renderRow(grp, specificGroups.length + i, true))}
                        </div>

                        {groups.length > 0 && (
                          <div className="mt-5 pt-4 border-t border-border/20 space-y-3">
                            <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest">
                              Top specific
                            </p>
                            {(() => {
                              // Show top specific classes first; fill remainder with generic if needed
                              const specifics = [...specificGroups].sort((a, b) => b.maxConf - a.maxConf);
                              const generics = [...genericGroups].sort((a, b) => b.maxConf - a.maxConf);
                              const top = [...specifics, ...generics].slice(0, 3);
                              return top.map((grp, i) => (
                                <div key={grp.label}>
                                  <div className="flex items-center justify-between mb-1">
                                    <span className={`text-[11px] truncate max-w-[120px] ${GENERIC.has(grp.label) ? "text-muted-foreground/50" : "text-foreground/70"}`}>
                                      {grp.label.replace(/_/g, " ")}
                                      {grp.count > 1 && (
                                        <span className="text-muted-foreground/50 ml-1">×{grp.count}</span>
                                      )}
                                    </span>
                                    <span className="text-[10px] font-mono text-muted-foreground">
                                      {(grp.maxConf * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                  <div className="h-1 rounded-full bg-secondary/60 overflow-hidden">
                                    <motion.div
                                      className="h-full rounded-full"
                                      initial={{ width: 0 }}
                                      animate={{ width: `${grp.maxConf * 100}%` }}
                                      transition={{ duration: 0.6, delay: i * 0.1, ease: "easeOut" }}
                                      style={{ backgroundColor: grp.color, opacity: GENERIC.has(grp.label) ? 0.5 : 1 }}
                                    />
                                  </div>
                                </div>
                              ));
                            })()}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              )}
            </div>
          ) : (
            <div className="p-5">
              <p className="text-[10px] font-semibold text-muted-foreground/50 uppercase tracking-widest mb-3">
                Analysis
              </p>
              <p className="text-xs text-muted-foreground">Select an image to view results.</p>
            </div>
          )}
        </div>
          </>
        )}
      </div>
    </div>
  );
};

export default DemoPage;