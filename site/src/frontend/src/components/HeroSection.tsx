import { Badge } from "@/components/ui/badge";
import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronDown,
  Copy,
  Trophy,
} from "lucide-react";
import {
  AnimatePresence,
  type Transition,
  type Variants,
  motion,
} from "motion/react";
import { useState } from "react";
import { toast } from "sonner";

// ─── Data ───────────────────────────────────────────────────────────────────

const STATS = [
  { value: "10", label: "Datasets" },
  { value: "8", label: "Baselines" },
  { value: "500k+", label: "Samples" },
  { value: "MIT", label: "License" },
];

const LEADERBOARD = [
  { rank: 1, model: "GNN", r2: "0.94", mae: "0.031" },
  { rank: 2, model: "LightGBM", r2: "0.91", mae: "0.044" },
  { rank: 3, model: "FCNN", r2: "0.89", mae: "0.058" },
  { rank: 4, model: "RandomForest", r2: "0.85", mae: "0.072" },
  { rank: 5, model: "LinearReg", r2: "0.71", mae: "0.124" },
];

// ─── Animation Variants ─────────────────────────────────────────────────────

const easeOut: Transition = { ease: "easeOut" };

const container: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.05 } },
};

const item: Variants = {
  hidden: { opacity: 0, y: 28 },
  show: { opacity: 1, y: 0, transition: { ...easeOut, duration: 0.6 } },
};

const heroAssetVariant: Variants = {
  hidden: { opacity: 0, y: 56 },
  show: {
    opacity: 1,
    y: 0,
    transition: { ...easeOut, duration: 0.8, delay: 0.55 },
  },
};

const statsVariant: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { ...easeOut, duration: 0.6, delay: 0.8 },
  },
};

// ─── Sub-components ─────────────────────────────────────────────────────────

function CodeEditorPane() {
  return (
    <div className="flex-1 min-w-0 flex flex-col bg-zinc-950/90 rounded-l-xl md:rounded-r-none rounded-r-xl">
      {/* MacOS title bar */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/8 bg-zinc-900/60 rounded-tl-xl rounded-tr-xl md:rounded-tr-none">
        <span className="w-3 h-3 rounded-full bg-red-500/80" />
        <span className="w-3 h-3 rounded-full bg-yellow-400/80" />
        <span className="w-3 h-3 rounded-full bg-green-500/80" />
        <span className="ml-auto text-xs text-zinc-500 font-mono">
          benchmark.py
        </span>
      </div>
      {/* Code */}
      <div className="overflow-x-auto p-4 flex-1">
        <pre className="font-mono text-xs sm:text-sm leading-relaxed whitespace-pre text-zinc-300">
          <span className="text-purple-400">from</span>
          <span className="text-zinc-300"> chembench.data.loader </span>
          <span className="text-purple-400">import</span>
          <span className="text-cyan-300"> ChemBenchDataLoader</span>
          {"\n"}
          <span className="text-purple-400">from</span>
          <span className="text-zinc-300"> chembench.models </span>
          <span className="text-purple-400">import</span>
          <span className="text-cyan-300"> RandomForestModel</span>
          {"\n\n"}
          <span className="text-zinc-500">
            # Load the ESOL solubility dataset
          </span>
          {"\n"}
          <span className="text-cyan-300">loader</span>
          <span className="text-zinc-300"> = </span>
          <span className="text-yellow-300">ChemBenchDataLoader</span>
          <span className="text-zinc-300">(</span>
          <span className="text-green-400">'esol'</span>
          <span className="text-zinc-300">)</span>
          {"\n"}
          <span className="text-cyan-300">X_train</span>
          <span className="text-zinc-300">, </span>
          <span className="text-cyan-300">X_test</span>
          <span className="text-zinc-300">, </span>
          <span className="text-cyan-300">y_train</span>
          <span className="text-zinc-300">, </span>
          <span className="text-cyan-300">y_test</span>
          <span className="text-zinc-300"> = loader.</span>
          <span className="text-yellow-300">get_splits</span>
          <span className="text-zinc-300">()</span>
          {"\n\n"}
          <span className="text-zinc-500"># Train a baseline model</span>
          {"\n"}
          <span className="text-cyan-300">model</span>
          <span className="text-zinc-300"> = </span>
          <span className="text-yellow-300">RandomForestModel</span>
          <span className="text-zinc-300">()</span>
          {"\n"}
          <span className="text-cyan-300">model</span>
          <span className="text-zinc-300">.</span>
          <span className="text-yellow-300">fit</span>
          <span className="text-zinc-300">(X_train, y_train)</span>
          {"\n"}
          <span className="text-cyan-300">metrics</span>
          <span className="text-zinc-300"> = model.</span>
          <span className="text-yellow-300">evaluate</span>
          <span className="text-zinc-300">(X_test, y_test)</span>
          {"\n"}
          <span className="text-purple-400">print</span>
          <span className="text-zinc-300">(metrics)</span>
        </pre>
      </div>
    </div>
  );
}

function LeaderboardPane() {
  return (
    <div className="flex-1 min-w-0 flex flex-col bg-zinc-950/90 border-t border-white/8 md:border-t-0 md:border-l rounded-b-xl md:rounded-b-none md:rounded-r-xl">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/8 bg-zinc-900/60 rounded-tr-none rounded-tl-none md:rounded-tr-xl">
        <Trophy className="w-3.5 h-3.5 text-yellow-400" />
        <span className="text-xs font-semibold text-zinc-200 tracking-wide">
          Benchmark Leaderboard
        </span>
        <span className="ml-auto text-xs text-zinc-500">ESOL · R²</span>
      </div>
      {/* Table */}
      <div className="p-3 flex-1">
        <div className="grid grid-cols-[auto_1fr_auto_auto] text-xs text-zinc-500 font-mono px-2 pb-1.5 gap-x-3">
          <span>#</span>
          <span>Model</span>
          <span>R²</span>
          <span>MAE</span>
        </div>
        {LEADERBOARD.map((row, i) => (
          <div
            key={row.rank}
            className={`grid grid-cols-[auto_1fr_auto_auto] items-center text-xs font-mono px-2 py-2 rounded-md gap-x-3 transition-colors ${
              i === 0
                ? "bg-cyan-500/10 border border-cyan-500/20 text-cyan-200"
                : "text-zinc-400 hover:bg-white/5"
            }`}
          >
            <span
              className={i === 0 ? "text-cyan-400 font-bold" : "text-zinc-600"}
            >
              {row.rank}
            </span>
            <span
              className={
                i === 0 ? "text-cyan-100 font-semibold" : "text-zinc-300"
              }
            >
              {row.model}
            </span>
            <span
              className={i === 0 ? "text-cyan-300 font-bold" : "text-zinc-400"}
            >
              {row.r2}
            </span>
            <span className="text-zinc-500">{row.mae}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function GlassmorphicHeroAsset() {
  return (
    <motion.div
      variants={heroAssetVariant}
      initial="hidden"
      animate="show"
      className="relative w-full max-w-4xl mx-auto mt-14"
      data-ocid="hero.floating_window"
    >
      {/* Outer glow */}
      <div
        className="absolute -inset-px rounded-xl pointer-events-none"
        style={{
          background:
            "linear-gradient(135deg, rgba(6,182,212,0.35) 0%, rgba(99,102,241,0.2) 50%, rgba(139,92,246,0.35) 100%)",
          borderRadius: "inherit",
          filter: "blur(1px)",
        }}
        aria-hidden="true"
      />
      {/* Inner container */}
      <motion.div
        animate={{ y: [0, -10, 0] }}
        transition={{
          duration: 5,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
        className="relative rounded-xl border border-white/10 overflow-hidden"
        style={{
          background: "rgba(9,9,11,0.85)",
          backdropFilter: "blur(24px)",
          boxShadow:
            "0 0 0 1px rgba(255,255,255,0.05), 0 32px 80px rgba(0,0,0,0.6), 0 0 40px rgba(6,182,212,0.12), 0 0 80px rgba(139,92,246,0.08)",
        }}
      >
        <div className="flex flex-col md:flex-row min-h-[280px]">
          <CodeEditorPane />
          <LeaderboardPane />
        </div>
      </motion.div>
    </motion.div>
  );
}

function GradientBorderButton({
  onClick,
  children,
}: {
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="relative group" data-ocid="hero.get_started_button">
      {/* Animated gradient border */}
      <div
        className="absolute -inset-[1.5px] rounded-xl opacity-70 group-hover:opacity-100 transition-opacity duration-300"
        style={{
          background:
            "conic-gradient(from var(--angle, 0deg), #06b6d4, #6366f1, #8b5cf6, #6366f1, #06b6d4)",
          animation: "spin-gradient 3s linear infinite",
        }}
        aria-hidden="true"
      />
      <button
        type="button"
        onClick={onClick}
        className="relative flex items-center gap-2 rounded-[10px] bg-zinc-950 px-7 py-3 text-sm font-semibold text-white transition-all duration-200 group-hover:[filter:drop-shadow(0_0_8px_rgba(6,182,212,0.7))]"
      >
        {children}
      </button>
    </div>
  );
}

function GlassButton({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      data-ocid="hero.docs_button"
      className="flex items-center gap-2 rounded-xl border border-white/20 bg-white/5 px-7 py-3 text-sm font-semibold text-white backdrop-blur-sm transition-all duration-200 hover:border-white/40 hover:bg-white/10 hover:shadow-[0_0_20px_rgba(255,255,255,0.06)]"
    >
      {children}
    </a>
  );
}

function CloneCopyBox() {
  const [copied, setCopied] = useState(false);
  const cloneCmd = "git clone https://github.com/IAyaanHere/ChemBench.git";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(cloneCmd);
      setCopied(true);
      toast.success("Copied to clipboard!");
      setTimeout(() => setCopied(false), 2500);
    } catch {
      toast.error("Failed to copy");
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label="Copy git clone command"
      data-ocid="hero.clone_snippet"
      className="group flex items-center gap-3 w-fit mx-auto cursor-pointer rounded-lg border border-zinc-700/80 bg-zinc-900 px-4 py-2.5 transition-all duration-200 hover:border-zinc-500 hover:scale-[1.015] hover:shadow-[0_4px_24px_rgba(6,182,212,0.12)] select-none"
    >
      <span className="font-mono text-zinc-500 text-sm">$</span>
      <span className="font-mono text-zinc-300 text-sm tracking-tight truncate max-w-[280px] sm:max-w-xs md:max-w-sm">
        {cloneCmd}
      </span>
      <span
        className="ml-2 flex-shrink-0 text-zinc-500 group-hover:text-zinc-300 transition-colors duration-150"
        data-ocid="hero.copy_button"
      >
        <AnimatePresence mode="wait" initial={false}>
          {copied ? (
            <motion.span
              key="check"
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.6, opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <Check className="w-4 h-4 text-cyan-400" />
            </motion.span>
          ) : (
            <motion.span
              key="copy"
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.6, opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <Copy className="w-4 h-4" />
            </motion.span>
          )}
        </AnimatePresence>
      </span>
    </button>
  );
}

function StatsBar() {
  return (
    <motion.div
      variants={statsVariant}
      initial="hidden"
      animate="show"
      className="w-fit mx-auto mt-10"
      data-ocid="hero.stats_bar"
    >
      <div
        className="flex flex-wrap justify-center gap-0 rounded-full border border-white/10 px-2 py-1"
        style={{
          background: "rgba(255,255,255,0.04)",
          backdropFilter: "blur(12px)",
          boxShadow:
            "0 8px 32px rgba(0,0,0,0.3), 0 0 40px rgba(6,182,212,0.05)",
        }}
      >
        {STATS.map((stat, i) => (
          <div
            key={stat.label}
            className={`flex flex-col items-center px-6 py-2.5 ${
              i < STATS.length - 1 ? "border-r border-white/10" : ""
            }`}
          >
            <span className="text-lg font-extrabold text-white tracking-tight leading-none">
              {stat.value}
            </span>
            <span className="text-xs text-zinc-500 mt-0.5 font-medium uppercase tracking-widest">
              {stat.label}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ─── Main Hero ───────────────────────────────────────────────────────────────

export default function HeroSection() {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden"
      style={{ background: "#09090B" }}
      data-ocid="hero.section"
    >
      {/* ── Animated radial gradient mesh ── */}
      <motion.div
        animate={{ scale: [1, 1.15, 1], opacity: [0.18, 0.28, 0.18] }}
        transition={{
          duration: 8,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
        className="absolute left-1/2 top-1/3 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[500px] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 55% 60% at 40% 50%, rgba(6,182,212,0.22) 0%, transparent 65%), radial-gradient(ellipse 50% 55% at 62% 45%, rgba(139,92,246,0.20) 0%, transparent 65%)",
          filter: "blur(48px)",
        }}
        aria-hidden="true"
      />
      <motion.div
        animate={{ scale: [1, 1.08, 1], opacity: [0.1, 0.18, 0.1] }}
        transition={{
          duration: 12,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
          delay: 4,
        }}
        className="absolute left-1/2 top-2/3 -translate-x-1/2 w-[500px] h-[300px] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 60% 60% at 50% 50%, rgba(99,102,241,0.18) 0%, transparent 70%)",
          filter: "blur(64px)",
        }}
        aria-hidden="true"
      />

      {/* ── Content ── */}
      <div className="relative z-10 w-full max-w-5xl mx-auto px-4 sm:px-6 text-center pt-28 pb-12">
        <motion.div variants={container} initial="hidden" animate="show">
          {/* Badge */}
          <motion.div variants={item} className="flex justify-center mb-7">
            <Badge
              variant="outline"
              className="px-4 py-1.5 text-xs font-medium border-cyan-500/30 text-cyan-400 bg-cyan-500/8 gap-2"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-glow-pulse inline-block" />
              Open Source · MIT License
            </Badge>
          </motion.div>

          {/* Headline */}
          <motion.h1
            variants={item}
            className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.05] text-white mb-6"
          >
            The Standardized{" "}
            <span
              className="inline-block"
              style={{
                background:
                  "linear-gradient(135deg, #22d3ee 0%, #818cf8 45%, #a855f7 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              ML Benchmark Suite
            </span>{" "}
            for Chemical Engineering
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            variants={item}
            className="text-lg sm:text-xl text-zinc-400 max-w-2xl mx-auto mb-9 leading-relaxed"
          >
            Bridging the gap between AI and Chemical Sciences. We provide the
            missing infrastructure: 10 curated datasets, 8 baseline models, and
            automated evaluation pipelines to accelerate your research.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            variants={item}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-7"
          >
            <GradientBorderButton onClick={() => scrollTo("quickstart")}>
              Get Started <ArrowRight className="w-4 h-4" />
            </GradientBorderButton>
            <GlassButton href="https://github.com/IAyaanHere/ChemBench">
              <BookOpen className="w-4 h-4" />
              Read Docs
            </GlassButton>
          </motion.div>

          {/* Clone copy box */}
          <motion.div variants={item}>
            <CloneCopyBox />
          </motion.div>
        </motion.div>

        {/* Glassmorphic hero asset */}
        <GlassmorphicHeroAsset />

        {/* Stats bar */}
        <StatsBar />
      </div>

      {/* Scroll indicator */}
      <button
        type="button"
        onClick={() => scrollTo("problem")}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 text-zinc-600 hover:text-zinc-300 transition-smooth animate-float"
        aria-label="Scroll down"
        data-ocid="hero.scroll_indicator"
      >
        <ChevronDown className="w-6 h-6" />
      </button>
    </section>
  );
}
