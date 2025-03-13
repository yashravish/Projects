import { View, Text, StyleSheet, ScrollView, Image, Pressable } from 'react-native';
import { Users, Star } from 'lucide-react-native';

const MATCHES = [
  {
    id: '1',
    name: 'Sarah Chen',
    role: 'Frontend Developer',
    compatibility: 95,
    skills: ['React', 'TypeScript', 'UI/UX'],
    image: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?q=80&w=1887&auto=format&fit=crop',
  },
  {
    id: '2',
    name: 'Michael Rodriguez',
    role: 'Full Stack Developer',
    compatibility: 88,
    skills: ['Node.js', 'React', 'MongoDB'],
    image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=1887&auto=format&fit=crop',
  },
  {
    id: '3',
    name: 'Emily Watson',
    role: 'Product Designer',
    compatibility: 85,
    skills: ['UI/UX', 'Figma', 'User Research'],
    image: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?q=80&w=1887&auto=format&fit=crop',
  },
];

export default function MatchesScreen() {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.title}>Your Matches</Text>
        <Text style={styles.subtitle}>Find your perfect team match</Text>
      </View>

      <View style={styles.matchesContainer}>
        {MATCHES.map((match) => (
          <Pressable key={match.id} style={styles.matchCard}>
            <View style={styles.matchHeader}>
              <Image source={{ uri: match.image }} style={styles.avatar} />
              <View style={styles.matchInfo}>
                <Text style={styles.name}>{match.name}</Text>
                <Text style={styles.role}>{match.role}</Text>
              </View>
              <View style={styles.compatibilityBadge}>
                <Star size={12} color="#fff" />
                <Text style={styles.compatibilityText}>{match.compatibility}%</Text>
              </View>
            </View>

            <View style={styles.skillsContainer}>
              {match.skills.map((skill) => (
                <View key={skill} style={styles.skillBadge}>
                  <Text style={styles.skillText}>{skill}</Text>
                </View>
              ))}
            </View>

            <Pressable style={styles.connectButton}>
              <Text style={styles.connectButtonText}>Connect</Text>
            </Pressable>
          </Pressable>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  content: {
    padding: 24,
  },
  header: {
    marginTop: 36,
    marginBottom: 24,
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
  matchesContainer: {
    gap: 16,
  },
  matchCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    gap: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  matchHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
  },
  matchInfo: {
    flex: 1,
  },
  name: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#111827',
  },
  role: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#6B7280',
  },
  compatibilityBadge: {
    backgroundColor: '#4F46E5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  compatibilityText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#fff',
  },
  skillsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  skillBadge: {
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  skillText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#4F46E5',
  },
  connectButton: {
    backgroundColor: '#4F46E5',
    padding: 12,
    borderRadius: 12,
    alignItems: 'center',
  },
  connectButtonText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
    color: '#fff',
  },
});