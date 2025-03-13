import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, ScrollView } from 'react-native';
import { router } from 'expo-router';
import { ArrowRight, Check } from 'lucide-react-native';

const SKILLS = [
  'React Native', 'React.js', 'Node.js', 'TypeScript',
  'Python', 'UI/UX Design', 'Product Management',
  'DevOps', 'Cloud Architecture', 'Data Science',
  'Machine Learning', 'Blockchain', 'iOS Development',
  'Android Development', 'Team Leadership',
];

export default function SkillsScreen() {
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);

  const toggleSkill = (skill: string) => {
    setSelectedSkills(prev =>
      prev.includes(skill)
        ? prev.filter(s => s !== skill)
        : [...prev, skill]
    );
  };

  const handleNext = () => {
    if (selectedSkills.length >= 3) {
      router.push('/(auth)/onboarding/personality');
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Select Your Skills</Text>
        <Text style={styles.subtitle}>Choose at least 3 skills that best describe your expertise</Text>
      </View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.skillsGrid}>
          {SKILLS.map((skill) => (
            <Pressable
              key={skill}
              style={[
                styles.skillBadge,
                selectedSkills.includes(skill) && styles.skillBadgeSelected
              ]}
              onPress={() => toggleSkill(skill)}
            >
              <Text style={[
                styles.skillText,
                selectedSkills.includes(skill) && styles.skillTextSelected
              ]}>
                {skill}
              </Text>
              {selectedSkills.includes(skill) && (
                <Check size={16} color="#fff" />
              )}
            </Pressable>
          ))}
        </View>
      </ScrollView>

      <View style={styles.footer}>
        <Text style={styles.counter}>
          {selectedSkills.length} of 3 minimum selected
        </Text>
        <Pressable
          style={[styles.button, selectedSkills.length < 3 && styles.buttonDisabled]}
          onPress={handleNext}
          disabled={selectedSkills.length < 3}
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
  skillsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  skillBadge: {
    backgroundColor: '#F3F4F6',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 100,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  skillBadgeSelected: {
    backgroundColor: '#4F46E5',
  },
  skillText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#4B5563',
  },
  skillTextSelected: {
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