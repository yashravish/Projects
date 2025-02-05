// src/screens/MoodTrackerScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import MoodTracker from '../components/MoodTracker';

export default function MoodTrackerScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <MoodTracker />
    </ScrollView>
  );
}
