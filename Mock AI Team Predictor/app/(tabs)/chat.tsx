import { View, Text, StyleSheet, ScrollView, Image, Pressable, TextInput } from 'react-native';
import { Search, MessageSquare } from 'lucide-react-native';

const CHATS = [
  {
    id: '1',
    name: 'Sarah Chen',
    lastMessage: 'Thanks for connecting! I\'d love to discuss potential collaboration.',
    time: '2m ago',
    unread: 2,
    image: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?q=80&w=1887&auto=format&fit=crop',
  },
  {
    id: '2',
    name: 'Michael Rodriguez',
    lastMessage: 'Looking forward to working together on the project!',
    time: '1h ago',
    unread: 0,
    image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=1887&auto=format&fit=crop',
  },
  {
    id: '3',
    name: 'Emily Watson',
    lastMessage: 'Let me know when you\'re free to chat about the design system.',
    time: '2h ago',
    unread: 1,
    image: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?q=80&w=1887&auto=format&fit=crop',
  },
];

export default function ChatScreen() {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Messages</Text>
        <View style={styles.searchContainer}>
          <Search size={20} color="#6B7280" />
          <TextInput
            style={styles.searchInput}
            placeholder="Search conversations"
            placeholderTextColor="#6B7280"
          />
        </View>
      </View>

      <ScrollView style={styles.chatList}>
        {CHATS.map((chat) => (
          <Pressable key={chat.id} style={styles.chatItem}>
            <Image source={{ uri: chat.image }} style={styles.avatar} />
            
            <View style={styles.chatContent}>
              <View style={styles.chatHeader}>
                <Text style={styles.name}>{chat.name}</Text>
                <Text style={styles.time}>{chat.time}</Text>
              </View>
              
              <View style={styles.messageContainer}>
                <Text 
                  style={styles.lastMessage}
                  numberOfLines={1}
                >
                  {chat.lastMessage}
                </Text>
                {chat.unread > 0 && (
                  <View style={styles.unreadBadge}>
                    <Text style={styles.unreadText}>{chat.unread}</Text>
                  </View>
                )}
              </View>
            </View>
          </Pressable>
        ))}
      </ScrollView>

      {CHATS.length === 0 && (
        <View style={styles.emptyState}>
          <MessageSquare size={48} color="#6B7280" />
          <Text style={styles.emptyTitle}>No messages yet</Text>
          <Text style={styles.emptyText}>
            Connect with your matches to start chatting
          </Text>
        </View>
      )}
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
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  title: {
    fontFamily: 'Inter-Bold',
    fontSize: 32,
    color: '#111827',
    marginBottom: 16,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    paddingHorizontal: 12,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    height: 44,
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#111827',
  },
  chatList: {
    flex: 1,
  },
  chatItem: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
  },
  chatContent: {
    flex: 1,
    gap: 4,
  },
  chatHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  name: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 16,
    color: '#111827',
  },
  time: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    color: '#6B7280',
  },
  messageContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  lastMessage: {
    flex: 1,
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#6B7280',
  },
  unreadBadge: {
    backgroundColor: '#4F46E5',
    width: 20,
    height: 20,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  unreadText: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 12,
    color: '#fff',
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    gap: 12,
  },
  emptyTitle: {
    fontFamily: 'Inter-SemiBold',
    fontSize: 18,
    color: '#111827',
  },
  emptyText: {
    fontFamily: 'Inter-Regular',
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
  },
});