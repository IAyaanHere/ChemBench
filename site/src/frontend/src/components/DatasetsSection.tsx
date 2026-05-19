import { Badge } from "@/components/ui/badge";
import { ChevronDown, Database } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

interface Dataset {
  name: string;
  samples: string;
  domain: string;
  taskType: string;
  rationale: string;
  source: string;
  color: string;
}

const DATASETS: Dataset[] = [
  {
    name: "Tennessee Eastman Process (TEP)",
    samples: "500,000+ samples",
    domain: "Industrial Process Monitoring",
    taskType: "Fault Detection Classification",
    rationale:
      "The industry standard benchmark for process monitoring and fault detection. Simulates a real chemical plant with 52 process variables and 21 fault types. Essential for validating process control ML methods.",
    source: "Harvard Dataverse",
    color: "#00d9ff",
  },
  {
    name: "QM9 — Quantum Chemistry",
    samples: "133,885 molecules",
    domain: "Quantum Chemistry",
    taskType: "Molecular Property Regression (12 targets)",
    rationale:
      "The gold standard benchmark for quantum chemistry ML. Contains DFT-computed properties (HOMO, LUMO, dipole moment, heat capacity) for 133k small organic molecules. Widely cited in graph neural network literature.",
    source: "Quantum Machine (qm9.py)",
    color: "#c855f7",
  },
  {
    name: "CheMixHub",
    samples: "500,000+ samples, 11 tasks",
    domain: "Mixture Thermodynamics",
    taskType: "Multi-task Regression",
    rationale:
      "Essential for modeling complex mixture properties and phase equilibria. Covers viscosity, density, vapor-liquid equilibrium, and activity coefficients across 11 distinct tasks. Uniquely relevant to ChemE process design.",
    source: "CheMixHub Repository",
    color: "#00d9ff",
  },
  {
    name: "ESOL (Solubility)",
    samples: "1,128 molecules",
    domain: "Physical Chemistry",
    taskType: "Aqueous Solubility Regression",
    rationale:
      "The canonical benchmark for molecular solubility prediction. Fundamental for drug formulation, environmental fate, and chemical process design. An interpretability testbed for explainable AI methods.",
    source: "MoleculeNet / Delaney 2004",
    color: "#c855f7",
  },
  {
    name: "FreeSolv (Hydration FE)",
    samples: "642 molecules",
    domain: "Computational Chemistry",
    taskType: "Hydration Free Energy Regression",
    rationale:
      "Experimental and computed hydration free energies for small molecules. Critical for validating molecular force fields and thermodynamic integration methods in computational chemistry.",
    source: "MoleculeNet / FreeSolv Database",
    color: "#00d9ff",
  },
  {
    name: "Lipophilicity",
    samples: "4,200 molecules",
    domain: "Drug ADME",
    taskType: "LogD Regression",
    rationale:
      "Octanol-water distribution coefficient (logD) predictions, a key ADME property for drug bioavailability. A real-world test for learned molecular representations in pharmaceutical applications.",
    source: "MoleculeNet / ChEMBL",
    color: "#c855f7",
  },
  {
    name: "HIV Antiviral Activity",
    samples: "41,127 molecules",
    domain: "Drug Discovery",
    taskType: "Binary Classification",
    rationale:
      "Large-scale HIV replication inhibition screening dataset. Tests models on highly imbalanced classification in a drug discovery context — directly relevant to ML-guided virtual screening pipelines.",
    source: "MoleculeNet / DTP AIDS",
    color: "#00d9ff",
  },
  {
    name: "Tox21 — Toxicity",
    samples: "7,831 molecules, 12 targets",
    domain: "Computational Toxicology",
    taskType: "Multi-label Classification",
    rationale:
      "Multi-label toxicity prediction across 12 nuclear receptor and stress response pathways. Essential for safe-by-design chemical synthesis and regulatory compliance. A true multi-task learning challenge.",
    source: "MoleculeNet / Tox21 Challenge",
    color: "#c855f7",
  },
];

export default function DatasetsSection() {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  return (
    <section
      id="datasets"
      className="py-24 bg-muted/20"
      data-ocid="datasets.section"
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            The Dataset{" "}
            <span
              style={{
                background: "linear-gradient(135deg, #00d9ff 0%, #c855f7 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              Collection
            </span>
          </h2>
          <p className="text-muted-foreground text-lg">
            8 carefully curated datasets spanning the full breadth of Chemical
            Engineering ML. Click any dataset to learn why we included it.
          </p>
        </motion.div>

        <div className="space-y-2" data-ocid="datasets.list">
          {DATASETS.map((ds, i) => (
            <motion.div
              key={ds.name}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
              data-ocid={`datasets.item.${i + 1}`}
            >
              <button
                type="button"
                onClick={() => setOpenIdx(openIdx === i ? null : i)}
                className="w-full flex items-center justify-between px-5 py-4 rounded-xl border transition-smooth text-left group"
                style={{
                  background:
                    openIdx === i
                      ? `linear-gradient(135deg, rgba(${ds.color === "#00d9ff" ? "0,217,255" : "200,85,247"},0.06) 0%, rgba(18,20,36,0.95) 100%)`
                      : "rgba(18,20,36,0.8)",
                  borderColor:
                    openIdx === i ? `${ds.color}40` : "rgba(255,255,255,0.08)",
                }}
                aria-expanded={openIdx === i}
                data-ocid={`datasets.toggle.${i + 1}`}
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{
                      background: `${ds.color}15`,
                      border: `1px solid ${ds.color}30`,
                    }}
                  >
                    <Database className="w-4 h-4" style={{ color: ds.color }} />
                  </div>
                  <span className="font-medium text-foreground text-sm sm:text-base truncate">
                    {ds.name}
                  </span>
                </div>
                <div className="flex items-center gap-3 ml-4 flex-shrink-0">
                  <Badge
                    variant="outline"
                    className="hidden sm:flex text-xs px-2 py-0.5 border-border text-muted-foreground"
                  >
                    {ds.samples}
                  </Badge>
                  <ChevronDown
                    className="w-4 h-4 text-muted-foreground transition-transform duration-300"
                    style={{
                      transform:
                        openIdx === i ? "rotate(180deg)" : "rotate(0deg)",
                    }}
                  />
                </div>
              </button>

              <AnimatePresence initial={false}>
                {openIdx === i && (
                  <motion.div
                    key="content"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: "easeInOut" }}
                    className="overflow-hidden"
                  >
                    <div
                      className="mx-1 mb-1 px-5 py-5 rounded-b-xl border-x border-b"
                      style={{
                        background: "rgba(10,12,22,0.95)",
                        borderColor: `${ds.color}25`,
                      }}
                    >
                      <div className="grid sm:grid-cols-3 gap-4 mb-4 text-sm">
                        <div>
                          <span className="text-xs text-muted-foreground uppercase tracking-widest block mb-1">
                            Domain
                          </span>
                          <span className="text-foreground font-medium">
                            {ds.domain}
                          </span>
                        </div>
                        <div>
                          <span className="text-xs text-muted-foreground uppercase tracking-widest block mb-1">
                            Task Type
                          </span>
                          <span className="text-foreground font-medium">
                            {ds.taskType}
                          </span>
                        </div>
                        <div>
                          <span className="text-xs text-muted-foreground uppercase tracking-widest block mb-1">
                            Source
                          </span>
                          <span className="text-foreground font-medium">
                            {ds.source}
                          </span>
                        </div>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground uppercase tracking-widest block mb-2">
                          Why We Included It
                        </span>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {ds.rationale}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
