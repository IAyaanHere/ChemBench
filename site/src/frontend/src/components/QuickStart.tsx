import { Check, Copy } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { toast } from "sonner";

type TokenType =
  | "keyword"
  | "class"
  | "module"
  | "string"
  | "func"
  | "comment"
  | "plain"
  | "builtin";
interface Token {
  id: string;
  t: TokenType;
  v: string;
}
interface CodeLine {
  lineId: string;
  tokens: Token[];
}

const HIGHLIGHTED_CODE: CodeLine[] = [
  {
    lineId: "L1",
    tokens: [
      { id: "1-0", t: "keyword", v: "from" },
      { id: "1-1", t: "plain", v: " " },
      { id: "1-2", t: "module", v: "chembench.data.loader" },
      { id: "1-3", t: "plain", v: " " },
      { id: "1-4", t: "keyword", v: "import" },
      { id: "1-5", t: "plain", v: " " },
      { id: "1-6", t: "class", v: "ChemBenchDataLoader" },
    ],
  },
  {
    lineId: "L2",
    tokens: [
      { id: "2-0", t: "keyword", v: "from" },
      { id: "2-1", t: "plain", v: " " },
      { id: "2-2", t: "module", v: "chembench.models" },
      { id: "2-3", t: "plain", v: " " },
      { id: "2-4", t: "keyword", v: "import" },
      { id: "2-5", t: "plain", v: " " },
      { id: "2-6", t: "class", v: "RandomForestModel" },
    ],
  },
  { lineId: "L3", tokens: [] },
  {
    lineId: "L4",
    tokens: [{ id: "4-0", t: "comment", v: "# Load standard dataset" }],
  },
  {
    lineId: "L5",
    tokens: [
      { id: "5-0", t: "plain", v: "loader" },
      { id: "5-1", t: "plain", v: " = " },
      { id: "5-2", t: "class", v: "ChemBenchDataLoader" },
      { id: "5-3", t: "plain", v: "(" },
      { id: "5-4", t: "string", v: "'esol'" },
      { id: "5-5", t: "plain", v: ")" },
    ],
  },
  {
    lineId: "L6",
    tokens: [
      { id: "6-0", t: "plain", v: "X_train, X_test, y_train, y_test" },
      { id: "6-1", t: "plain", v: " = " },
      { id: "6-2", t: "plain", v: "loader." },
      { id: "6-3", t: "func", v: "get_splits" },
      { id: "6-4", t: "plain", v: "()" },
    ],
  },
  { lineId: "L7", tokens: [] },
  {
    lineId: "L8",
    tokens: [
      { id: "8-0", t: "comment", v: "# Train and evaluate automatically" },
    ],
  },
  {
    lineId: "L9",
    tokens: [
      { id: "9-0", t: "plain", v: "model" },
      { id: "9-1", t: "plain", v: " = " },
      { id: "9-2", t: "class", v: "RandomForestModel" },
      { id: "9-3", t: "plain", v: "()" },
    ],
  },
  {
    lineId: "L10",
    tokens: [
      { id: "10-0", t: "plain", v: "model." },
      { id: "10-1", t: "func", v: "fit" },
      { id: "10-2", t: "plain", v: "(X_train, y_train)" },
    ],
  },
  {
    lineId: "L11",
    tokens: [
      { id: "11-0", t: "builtin", v: "print" },
      { id: "11-1", t: "plain", v: "(model." },
      { id: "11-2", t: "func", v: "evaluate" },
      { id: "11-3", t: "plain", v: "(X_test, y_test))" },
    ],
  },
];

const TOKEN_COLORS: Record<TokenType, string> = {
  keyword: "#c855f7",
  class: "#00d9ff",
  module: "#a8d8ff",
  string: "#a8ff82",
  func: "#ffd080",
  comment: "#6e7e8a",
  plain: "#e2e8f0",
  builtin: "#00d9ff",
};

const RAW_CODE = `from chembench.data.loader import ChemBenchDataLoader
from chembench.models import RandomForestModel

# Load standard dataset
loader = ChemBenchDataLoader('esol')
X_train, X_test, y_train, y_test = loader.get_splits()

# Train and evaluate automatically
model = RandomForestModel()
model.fit(X_train, y_train)
print(model.evaluate(X_test, y_test))`;

export default function QuickStart() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(RAW_CODE);
      setCopied(true);
      toast.success("Code copied to clipboard!");
      setTimeout(() => setCopied(false), 2500);
    } catch {
      toast.error("Failed to copy");
    }
  };

  const renderTokens = (tokens: Token[]) =>
    tokens.map((tok) => (
      <span key={tok.id} style={{ color: TOKEN_COLORS[tok.t] }}>
        {tok.v}
      </span>
    ));

  return (
    <section
      id="quickstart"
      className="py-24 bg-background"
      data-ocid="quickstart.section"
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Get Up and Running in{" "}
            <span
              style={{
                background: "linear-gradient(135deg, #00d9ff 0%, #c855f7 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              Minutes
            </span>
          </h2>
          <p className="text-muted-foreground text-lg">
            Install ChemBench, load a dataset, and run your first evaluation
            with a single baseline model.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="code-window w-full"
          data-ocid="quickstart.code_window"
        >
          {/* macOS header */}
          <div className="code-header">
            <span className="code-dot bg-red-500" />
            <span className="code-dot bg-yellow-500" />
            <span className="code-dot bg-green-500" />
            <span className="ml-4 text-xs text-muted-foreground font-mono flex-1">
              chembench_quickstart.py
            </span>
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-smooth px-2 py-1 rounded hover:bg-primary/10"
              aria-label="Copy code"
              data-ocid="quickstart.copy_button"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-primary" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy
                </>
              )}
            </button>
          </div>

          {/* Code body */}
          <div className="overflow-x-auto" data-ocid="quickstart.code_body">
            <pre
              className="px-5 py-5 text-sm leading-relaxed font-mono"
              style={{ minWidth: "max-content" }}
            >
              {HIGHLIGHTED_CODE.map((line, li) => (
                <div key={line.lineId} className="flex">
                  <span
                    className="select-none mr-5 text-right"
                    style={{ color: "#3a4558", minWidth: "1.5rem" }}
                  >
                    {li + 1}
                  </span>
                  <span>
                    {line.tokens.length === 0 ? (
                      <>&nbsp;</>
                    ) : (
                      <>{renderTokens(line.tokens)}</>
                    )}
                  </span>
                </div>
              ))}
            </pre>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4 text-sm text-muted-foreground"
        >
          <span>Install with:</span>
          <code className="bg-card border border-border px-3 py-1.5 rounded-lg font-mono text-primary text-xs">
            pip install git+https://github.com/IAyaanHere/ChemBench.git
          </code>
          <span>or</span>
          <a
            href="https://github.com/IAyaanHere/ChemBench"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline underline-offset-4"
            data-ocid="quickstart.github_link"
          >
            View full docs on GitHub →
          </a>
        </motion.div>
      </div>
    </section>
  );
}
