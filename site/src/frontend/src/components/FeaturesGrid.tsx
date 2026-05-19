import { BarChart2, Cpu, GitFork, Globe } from "lucide-react";
import { motion } from "motion/react";

const FEATURES = [
  {
    icon: GitFork,
    title: "Standardized Data Pipeline",
    description:
      "Built-in scaffold splitting, missing value imputation, outlier detection, and SMILES validation. Every dataset ready to use in minutes.",
    gradient: "from-cyan-500/10 to-blue-500/10",
    border: "rgba(0,217,255,0.2)",
    glow: "rgba(0,217,255,0.15)",
    iconColor: "#00d9ff",
  },
  {
    icon: Cpu,
    title: "8 Ready-to-Use Baselines",
    description:
      "Scikit-Learn models (Linear Regression, Random Forest, GBM, LightGBM) and PyTorch deep learning (FCNN, CNN1D, GNN). Plug-and-play.",
    gradient: "from-purple-500/10 to-pink-500/10",
    border: "rgba(200,85,247,0.2)",
    glow: "rgba(200,85,247,0.15)",
    iconColor: "#c855f7",
  },
  {
    icon: BarChart2,
    title: "Automated Leaderboards",
    description:
      "Instantly generate publication-quality reports, performance graphs, and comparison tables. From raw results to PDF in seconds.",
    gradient: "from-emerald-500/10 to-cyan-500/10",
    border: "rgba(0,217,255,0.2)",
    glow: "rgba(0,217,255,0.15)",
    iconColor: "#00d9ff",
  },
  {
    icon: Globe,
    title: "100% Open Source",
    description:
      "MIT Licensed. Fully transparent, community-driven, and built for reproducibility. Fork it, extend it, improve it — no strings attached.",
    gradient: "from-purple-500/10 to-indigo-500/10",
    border: "rgba(200,85,247,0.2)",
    glow: "rgba(200,85,247,0.15)",
    iconColor: "#c855f7",
  },
];

export default function FeaturesGrid() {
  return (
    <section
      id="solution"
      className="py-24 bg-background"
      data-ocid="features.section"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Built for{" "}
            <span
              style={{
                background: "linear-gradient(135deg, #00d9ff 0%, #c855f7 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              Reproducible Research
            </span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Every component of ChemBench is designed to eliminate friction from
            your ML research workflow.
          </p>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURES.map((feat, i) => (
            <motion.div
              key={feat.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              whileHover={{ y: -6, boxShadow: `0 0 30px ${feat.glow}` }}
              className="rounded-2xl border p-6 transition-smooth cursor-default"
              style={{
                background:
                  "linear-gradient(135deg, rgba(18,20,36,0.95) 0%, rgba(14,16,28,0.95) 100%)",
                borderColor: feat.border,
              }}
              data-ocid={`features.card.${i + 1}`}
            >
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center mb-5"
                style={{
                  background: `linear-gradient(135deg, ${feat.glow} 0%, transparent 100%)`,
                  border: `1px solid ${feat.border}`,
                }}
              >
                <feat.icon
                  className="w-6 h-6"
                  style={{ color: feat.iconColor }}
                />
              </div>
              <h3 className="font-semibold text-foreground mb-3 text-base leading-snug">
                {feat.title}
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {feat.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
