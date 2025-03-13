import { useState } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TextInput, 
  Pressable, 
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  TouchableWithoutFeedback,
  Keyboard
} from 'react-native';
import { router } from 'expo-router';
import { ArrowRight } from 'lucide-react-native';

export default function GoalsScreen() {
  const [shortTerm, setShortTerm] = useState('');
  const [longTerm, setLongTerm] = useState('');
  const [interests, setInterests] = useState('');

  const handleNext = () => {
    if (shortTerm && longTerm && interests) {
      router.push('/(auth)/onboarding/analyzing');
    }
  };

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <View style={styles.container}>
          <View style={styles.header}>
            <Text style={styles.title}>Your Goals & Interests</Text>
            <Text style={styles.subtitle}>Help us understand what you're looking for</Text>
          </View>

          <ScrollView 
            style={styles.content} 
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
          >
            <View style={styles.form}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>What are your short-term career goals?</Text>
                <TextInput
                  style={styles.input}
                  value={shortTerm}
                  onChangeText={setShortTerm}
                  placeholder="e.g., Learn new technologies, lead a team"
                  placeholderTextColor="#94A3B8"
                  multiline
                  numberOfLines={3}
                  returnKeyType="next"
                  blurOnSubmit={true}
                />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>What are your long-term career aspirations?</Text>
                <TextInput
                  style={styles.input}
                  value={longTerm}
                  onChangeText={setLongTerm}
                  placeholder="e.g., Start a company, become a technical architect"
                  placeholderTextColor="#94A3B8"
                  multiline
                  numberOfLines={3}
                  returnKeyType="next"
                  blurOnSubmit={true}
                />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>What kind of projects interest you most?</Text>
                <TextInput
                  style={styles.input}
                  value={interests}
                  onChangeText={setInterests}
                  placeholder="e.g., AI/ML projects, mobile apps, enterprise software"
                  placeholderTextColor="#94A3B8"
                  multiline
                  numberOfLines={3}
                  returnKeyType="done"
                  blurOnSubmit={true}
                />
              </View>
            </View>

            <View style={styles.footer}>
              <Pressable
                style={[
                  styles.button,
                  (!shortTerm || !longTerm || !interests) && styles.buttonDisabled
                ]}
                onPress={handleNext}
                disabled={!shortTerm || !longTerm || !interests}
              >
                <Text style={styles.buttonText}>Continue</Text>
                <ArrowRight size={20} color="#fff" />
              </Pressable>
            </View>
          </ScrollView>
        </View>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
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
  },
  form: {
    padding: 24,
    gap: 24,
  },
  inputGroup: {
    gap: 8,
  },
  label: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#111827',
  },
  input: {
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#111827',
    fontFamily: 'Inter-Regular',
    minHeight: 100,
    textAlignVertical: 'top',
  },
  footer: {
    padding: 24,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
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