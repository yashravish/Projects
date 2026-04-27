import { Hero } from '@/components/landing/hero';
import { FeatureShowcase } from '@/components/landing/feature-showcase';
import { CTASection } from '@/components/landing/cta-section';

export default function LandingPage() {
  return (
    <>
      <Hero />
      <FeatureShowcase />
      <CTASection />
    </>
  );
}
