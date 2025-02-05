// src/screens/MeditationScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import MeditationPlayer from '../components/MeditationPlayer';

export default function MeditationScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <MeditationPlayer />
    </ScrollView>
  );
}
