// src/screens/JournalScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import JournalPrompt from '../components/JournalPrompt';

export default function JournalScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <JournalPrompt />
    </ScrollView>
  );
}
