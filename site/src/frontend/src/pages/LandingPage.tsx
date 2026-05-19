import DatasetsSection from "@/components/DatasetsSection";
import FeaturesGrid from "@/components/FeaturesGrid";
import Footer from "@/components/Footer";
import HeroSection from "@/components/HeroSection";
import Navbar from "@/components/Navbar";
import ProblemSolution from "@/components/ProblemSolution";
import QuickStart from "@/components/QuickStart";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <main>
        <HeroSection />
        <ProblemSolution />
        <FeaturesGrid />
        <DatasetsSection />
        <QuickStart />
      </main>
      <Footer />
    </div>
  );
}
