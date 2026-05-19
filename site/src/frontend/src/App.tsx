import { Toaster } from "@/components/ui/sonner";
import LandingPage from "@/pages/LandingPage";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
        <LandingPage />
        <Toaster />
      </ThemeProvider>
    </QueryClientProvider>
  );
}
