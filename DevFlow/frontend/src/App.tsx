import { Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { Dashboard } from "@/pages/Dashboard";
import { Projects } from "@/pages/Projects";
import { Pipelines } from "@/pages/Pipelines";
import { Deployments } from "@/pages/Deployments";
import { Flags } from "@/pages/Flags";
import { Experiments } from "@/pages/Experiments";
import { Defects } from "@/pages/Defects";
import { Knowledge } from "@/pages/Knowledge";
import { AIAnalyzer } from "@/pages/AIAnalyzer";
import { MetricsPage } from "@/pages/MetricsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/pipelines" element={<Pipelines />} />
        <Route path="/deployments" element={<Deployments />} />
        <Route path="/flags" element={<Flags />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/defects" element={<Defects />} />
        <Route path="/knowledge" element={<Knowledge />} />
        <Route path="/ai" element={<AIAnalyzer />} />
        <Route path="/metrics" element={<MetricsPage />} />
      </Route>
    </Routes>
  );
}
