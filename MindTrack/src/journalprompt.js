// src/components/JournalPrompt.js
import React, { useState, useContext } from 'react';
import { View, Text, TextInput, Pressable } from 'react-native';
import { WellnessContext } from '../context/WellnessContext';

export default function JournalPrompt() {
  const { addJournalEntry } = useContext(WellnessContext);
  const [entry, setEntry] = useState('');

  const handleSubmit = () => {
    if (entry.trim()) {
      addJournalEntry({ date: new Date(), text: entry });
      setEntry('');
    }
  };

  return (
    <View className="p-4">
      <Text className="text-xl font-bold mb-4">Today's Journal Prompt:</Text>
      <Text className="mb-2">What are you grateful for today?</Text>
      <TextInput
        className="border border-gray-300 p-2 rounded mb-4"
        placeholder="Write your thoughts..."
        value={entry}
        onChangeText={setEntry}
        multiline
      />
      <Pressable onPress={handleSubmit} className="bg-green-500 py-2 px-4 rounded">
        <Text className="text-white text-center">Submit Entry</Text>
      </Pressable>
    </View>
  );
}
