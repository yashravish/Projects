import { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { router } from 'expo-router';
import { Brain } from 'lucide-react-native';
import Animated, {
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
  cancelAnimation,
  useSharedValue,
} from 'react-native-reanimated';

const ANALYSIS_STEPS = [
  'Analyzing your skills and experience...',
  'Processing your work style preferences...',
  'Evaluating team compatibility factors...',
  'Generating your AI compatibility profile...',
];

export default function AnalyzingScreen() {
  const rotation = useSharedValue(0);
  const [currentStep, setCurrentStep] = useState(0);
  
  useEffect(() => {
    rotation.value = withRepeat(
      withSequence(
        withTiming(360, { duration: 2000 }),
      ),
      -1
    );

    // Progress through steps
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev < ANALYSIS_STEPS.length - 1) return prev + 1;
        clearInterval(stepInterval);
        return prev;
      });
    }, 1500);

    // Navigate to tabs after all steps
    const timer = setTimeout(() => {
      cancelAnimation(rotation);
      router.replace('/(tabs)');
    }, 6000);

    return () => {
      clearTimeout(timer);
      clearInterval(stepInterval);
      cancelAnimation(rotation);
    };
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${rotation.value}deg` }],
  }));

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Animated.View style={animatedStyle}>
          <Brain size={64} color="#4F46E5" />
        </Animated.View>
        
        <View style={styles.textContainer}>
          <Text style={styles.title}>Creating Your Profile</Text>
          <Text style={styles.subtitle}>Our AI is analyzing your responses</Text>
        </View>

        <View style={styles.steps}>
          {ANALYSIS_STEPS.map((step, index) => (
            <Text 
              key={index} 
              style={[
                styles.step,
                index === currentStep && styles.activeStep,
                index < currentStep && styles.completedStep
              ]}
            >
              {step}
            </Text>
          ))}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  content: {
    alignItems: 'center',
    gap: 24,
  },
  textContainer: {
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontFamily: 'Inter-Bold',
    fontSize: 24,
    color: '#111827',
    textAlign: 'center',
  },
  subtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#6B7280',
    textAlign: 'center',
  },
  steps: {
    gap: 16,
    marginTop: 48,
  },
  step: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    opacity: 0.5,
  },
  activeStep: {
    color: '#4F46E5',
    opacity: 1,
    fontFamily: 'Inter-SemiBold',
  },
  completedStep: {
    color: '#10B981',
    opacity: 1,
  },
});