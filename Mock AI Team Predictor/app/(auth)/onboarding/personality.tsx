import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, ScrollView } from 'react-native';
import { router } from 'expo-router';
import { ArrowRight } from 'lucide-react-native';

const QUESTIONS = [
  {
    id: 'work_style',
    question: 'How do you prefer to work?',
    options: [
      { id: 'independent', text: 'Independently, with freedom to make decisions' },
      { id: 'collaborative', text: 'Collaboratively, as part of a team' },
      { id: 'mixed', text: 'A mix of both, depending on the task' },
    ],
  },
  {
    id: 'communication',
    question: 'What\'s your communication style?',
    options: [
      { id: 'direct', text: 'Direct and to the point' },
      { id: 'diplomatic', text: 'Diplomatic and considerate' },
      { id: 'casual', text: 'Casual and friendly' },
    ],
  },
  {
    id: 'problem_solving',
    question: 'How do you approach problems?',
    options: [
      { id: 'analytical', text: 'Analytical and methodical' },
      { id: 'creative', text: 'Creative and intuitive' },
      { id: 'practical', text: 'Practical and results-oriented' },
    ],
  },
];

export default function PersonalityScreen() {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleAnswer = (questionId: string, optionId: string) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: optionId,
    }));
  };

  const handleNext = () => {
    if (Object.keys(answers).length === QUESTIONS.length) {
      router.push('/(auth)/onboarding/goals');
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Your Work Style</Text>
        <Text style={styles.subtitle}>Help us understand how you work best</Text>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {QUESTIONS.map((question) => (
          <View key={question.id} style={styles.questionContainer}>
            <Text style={styles.questionText}>{question.question}</Text>
            <View style={styles.optionsContainer}>
              {question.options.map((option) => (
                <Pressable
                  key={option.id}
                  style={[
                    styles.optionButton,
                    answers[question.id] === option.id && styles.optionButtonSelected
                  ]}
                  onPress={() => handleAnswer(question.id, option.id)}
                >
                  <Text style={[
                    styles.optionText,
                    answers[question.id] === option.id && styles.optionTextSelected
                  ]}>
                    {option.text}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
        ))}
      </ScrollView>

      <View style={styles.footer}>
        <Text style={styles.counter}>
          {Object.keys(answers).length} of {QUESTIONS.length} questions answered
        </Text>
        <Pressable
          style={[
            styles.button,
            Object.keys(answers).length < QUESTIONS.length && styles.buttonDisabled
          ]}
          onPress={handleNext}
          disabled={Object.keys(answers).length < QUESTIONS.length}
        >
          <Text style={styles.buttonText}>Continue</Text>
          <ArrowRight size={20} color="#fff" />
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    padding: 24,
    paddingTop: 60,
    backgroundColor: '#fff',
  },
  title: {
    fontFamily: 'Inter-Bold',
    fontSize: 32,
    color: '#111827',
    marginBottom: 8,
  },
  subtitle: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#6B7280',
  },
  content: {
    flex: 1,
    padding: 24,
  },
  questionContainer: {
    marginBottom: 32,
  },
  questionText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 18,
    color: '#111827',
    marginBottom: 16,
  },
  optionsContainer: {
    gap: 12,
  },
  optionButton: {
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#F3F4F6',
  },
  optionButtonSelected: {
    backgroundColor: '#4F46E5',
  },
  optionText: {
    fontFamily: 'Inter-Regular',
    fontSize: 16,
    color: '#4B5563',
  },
  optionTextSelected: {
    color: '#fff',
  },
  footer: {
    padding: 24,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
  },
  counter: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#4F46E5',
    padding: 16,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 18,
    color: '#fff',
  },
});