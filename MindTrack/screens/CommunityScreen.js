// src/screens/CommunityScreen.js
import React from 'react';
import { ScrollView } from 'react-native';
import CommunityGroup from '../components/CommunityGroup';

export default function CommunityScreen() {
  return (
    <ScrollView className="flex-1 bg-white p-4">
      <CommunityGroup />
    </ScrollView>
  );
}
