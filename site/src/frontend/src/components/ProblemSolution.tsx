import { AlertTriangle, Zap } from "lucide-react";
import { motion } from "motion/react";

const fadeInLeft = {
  hidden: { opacity: 0, x: -32 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6 } },
};
const fadeInRight = {
  hidden: { opacity: 0, x: 32 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6 } },
};

export default function ProblemSolution() {
  return (
    <section
      id="problem"
      className="py-24 bg-muted/20"
      data-ocid="problem.section"
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
            The Challenge &{" "}
            <span
              style={{
                background: "linear-gradient(135deg, #00d9ff 0%, #c855f7 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              Our Answer
            </span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Chemical Engineering research has a reproducibility problem.
            ChemBench fixes it.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Problem card */}
          <motion.div
            variants={fadeInLeft}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="group rounded-2xl border p-8 transition-smooth cursor-default"
            style={{
              background:
                "linear-gradient(135deg, rgba(30,10,10,0.9) 0%, rgba(20,12,12,0.9) 100%)",
              borderColor: "rgba(239, 68, 68, 0.25)",
              boxShadow: "0 0 0 0 rgba(239,68,68,0)",
            }}
            whileHover={{
              boxShadow:
                "0 0 30px rgba(239,68,68,0.18), 0 0 60px rgba(239,68,68,0.1)",
              borderColor: "rgba(239, 68, 68, 0.45)",
              y: -4,
            }}
            data-ocid="problem.problem_card"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-destructive/15 border border-destructive/30 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-destructive" />
              </div>
              <span className="text-xs font-semibold uppercase tracking-widest text-destructive/80">
                The Problem
              </span>
            </div>
            <h3 className="text-2xl font-bold text-foreground mb-4">
              Why Are We Lagging?
            </h3>
            <p className="text-muted-foreground leading-relaxed mb-6">
              Machine Learning is transforming every scientific field — but
              Chemical Engineering is falling behind. The root causes are
              systemic:
            </p>
            <ul className="space-y-3 text-sm text-muted-foreground">
              {[
                "Datasets are scattered across dozens of incompatible repositories",
                "No unified benchmarks to measure and compare model performance",
                "Comparing ML methods requires months of custom preprocessing work",
                "Reproduced results rarely match original papers due to protocol differences",
              ].map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="text-destructive mt-1 flex-shrink-0">✗</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </motion.div>

          {/* Solution card */}
          <motion.div
            variants={fadeInRight}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="group rounded-2xl border p-8 transition-smooth cursor-default"
            style={{
              background:
                "linear-gradient(135deg, rgba(0,25,30,0.9) 0%, rgba(0,20,25,0.9) 100%)",
              borderColor: "rgba(0, 217, 255, 0.20)",
              boxShadow: "0 0 0 0 rgba(0,217,255,0)",
            }}
            whileHover={{
              boxShadow:
                "0 0 30px rgba(0,217,255,0.18), 0 0 60px rgba(0,217,255,0.1)",
              borderColor: "rgba(0, 217, 255, 0.45)",
              y: -4,
            }}
            data-ocid="problem.solution_card"
          >
            <div className="flex items-center gap-3 mb-6">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{
                  background: "rgba(0,217,255,0.12)",
                  border: "1px solid rgba(0,217,255,0.3)",
                }}
              >
                <Zap className="w-5 h-5" style={{ color: "#00d9ff" }} />
              </div>
              <span
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "#00d9ff99" }}
              >
                The Solution
              </span>
            </div>
            <h3 className="text-2xl font-bold text-foreground mb-4">
              The ChemBench Solution
            </h3>
            <p className="text-muted-foreground leading-relaxed mb-6">
              ChemBench provides the ultimate unified infrastructure for ML
              research in Chemical Engineering — from raw data to
              publication-ready leaderboards.
            </p>
            <ul className="space-y-3 text-sm text-muted-foreground">
              {[
                "10 curated datasets spanning molecular, process, and mixture properties",
                "Standardized preprocessing, scaffold splitting, and SMILES validation",
                "8 baseline models from Random Forest to Graph Neural Networks",
                "Automated leaderboard generation with publication-quality reports",
              ].map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span
                    className="mt-1 flex-shrink-0"
                    style={{ color: "#00d9ff" }}
                  >
                    ✓
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
