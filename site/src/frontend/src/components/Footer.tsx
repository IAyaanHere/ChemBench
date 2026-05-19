import { Bug, FileText, FlaskConical, Github } from "lucide-react";

export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer
      className="bg-card border-t border-border"
      data-ocid="footer.section"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          {/* Brand */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-primary/20 border border-primary/40 flex items-center justify-center">
                <FlaskConical className="w-3.5 h-3.5 text-primary" />
              </div>
              <span className="font-display font-bold text-lg text-foreground">
                Chem<span className="text-primary">Bench</span>
              </span>
            </div>
            <p className="text-sm text-muted-foreground max-w-xs leading-relaxed">
              Made with love by ChemBench
            </p>
          </div>

          {/* Links */}
          <nav className="flex flex-wrap gap-6" aria-label="Footer navigation">
            <a
              href="https://github.com/IAyaanHere/ChemBench"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary transition-smooth"
              data-ocid="footer.github_link"
            >
              <Github className="w-4 h-4" />
              GitHub
            </a>
            <a
              href="https://github.com/IAyaanHere/ChemBench/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary transition-smooth"
              data-ocid="footer.issues_link"
            >
              <Bug className="w-4 h-4" />
              Report an Issue
            </a>
            <a
              href="https://github.com/IAyaanHere/ChemBench/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary transition-smooth"
              data-ocid="footer.license_link"
            >
              <FileText className="w-4 h-4" />
              License (MIT)
            </a>
          </nav>
        </div>

        <div className="mt-10 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-muted-foreground">
          <span>© {year} ChemBench. All rights reserved.</span>
          <span>Built with love by ChemBench</span>
        </div>
      </div>
    </footer>
  );
}
