// src/screens/DirectoryScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import ProfessionalDirectory from '../components/ProfessionalDirectory';

export default function DirectoryScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <ProfessionalDirectory />
    </ScrollView>
  );
}
