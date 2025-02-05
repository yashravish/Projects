// src/screens/EmergencyScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import EmergencyLocator from '../components/EmergencyLocator';

export default function EmergencyScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <EmergencyLocator />
    </ScrollView>
  );
}
