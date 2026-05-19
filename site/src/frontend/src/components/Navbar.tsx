import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { FlaskConical, Github, Menu, Star, X } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";

const navLinks = [
  { label: "The Problem", href: "#problem" },
  { label: "The Solution", href: "#solution" },
  { label: "Datasets", href: "#datasets" },
  { label: "Quick Start", href: "#quickstart" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollTo = (href: string) => {
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: "smooth" });
    setMobileOpen(false);
  };

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        scrolled
          ? "bg-zinc-950/95 backdrop-blur-md border-b border-white/8 shadow-lg shadow-black/40"
          : "bg-transparent",
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <button
            type="button"
            className="flex items-center gap-2 group"
            onClick={() => {
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            data-ocid="nav.logo"
          >
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center group-hover:bg-cyan-500/20 transition-all duration-200">
              <FlaskConical className="w-4 h-4 text-cyan-400" />
            </div>
            <span className="font-display font-bold text-xl text-white tracking-tight">
              Chem<span className="text-cyan-400">Bench</span>
            </span>
          </button>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <button
                key={link.href}
                type="button"
                onClick={() => scrollTo(link.href)}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors duration-200 rounded-md hover:bg-white/5"
                data-ocid={`nav.${link.label.toLowerCase().replace(/\s+/g, "-")}.link`}
              >
                {link.label}
              </button>
            ))}
          </nav>

          {/* GitHub CTA */}
          <div className="hidden md:flex items-center gap-3">
            <a
              href="https://github.com/IAyaanHere/ChemBench"
              target="_blank"
              rel="noopener noreferrer"
              data-ocid="nav.github_button"
              className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-1.5 text-sm font-medium text-zinc-200 transition-all duration-200 hover:border-zinc-500 hover:bg-zinc-800 hover:text-white"
            >
              <Github className="w-4 h-4" />
              <Star className="w-3 h-3 fill-current text-yellow-400" />
              Star on GitHub
            </a>
          </div>

          {/* Mobile hamburger */}
          <button
            type="button"
            className="md:hidden p-2 text-zinc-500 hover:text-white transition-colors"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label={mobileOpen ? "Close menu" : "Open menu"}
            data-ocid="nav.mobile_menu_toggle"
          >
            {mobileOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden bg-zinc-950/98 backdrop-blur-md border-b border-white/8">
          <div className="px-4 py-4 space-y-1">
            {navLinks.map((link) => (
              <button
                key={link.href}
                type="button"
                onClick={() => scrollTo(link.href)}
                className="block w-full text-left px-3 py-2 text-sm text-zinc-400 hover:text-white hover:bg-white/5 rounded-md transition-colors"
                data-ocid={`nav.mobile_${link.label.toLowerCase().replace(/\s+/g, "_")}.link`}
              >
                {link.label}
              </button>
            ))}
            <a
              href="https://github.com/IAyaanHere/ChemBench"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-2 text-sm text-cyan-400 font-medium"
              data-ocid="nav.mobile_github_button"
            >
              <Github className="w-4 h-4" />
              Star on GitHub
            </a>
          </div>
        </div>
      )}
    </motion.header>
  );
}
