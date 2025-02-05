// src/screens/ProgressScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import ProgressChart from '../components/ProgressChart';

export default function ProgressScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <ProgressChart />
    </ScrollView>
  );
}
