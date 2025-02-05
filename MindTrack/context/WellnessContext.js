// src/context/WellnessContext.js
import React, { createContext, useState } from 'react';

export const WellnessContext = createContext();

export const WellnessProvider = ({ children }) => {
  const [moodEntries, setMoodEntries] = useState([]);
  const [journalEntries, setJournalEntries] = useState([]);

  const addMoodEntry = (entry) => {
    setMoodEntries((prev) => [...prev, entry]);
  };

  const addJournalEntry = (entry) => {
    setJournalEntries((prev) => [...prev, entry]);
  };

  return (
    <WellnessContext.Provider
      value={{ moodEntries, addMoodEntry, journalEntries, addJournalEntry }}
    >
      {children}
    </WellnessContext.Provider>
  );
};
