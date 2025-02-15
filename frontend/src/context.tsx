import React, { createContext, useContext, useState, ReactNode, useEffect } from "react";

// Define the user type
interface User {
  id: string;
  name: string;
  email: string;
}

// Define the context type
interface CurrConfigContextType {
  user: User | null;
  setUser: (user: User | null) => void;
}

// Create the context
export const CurrConfigContext = createContext<CurrConfigContextType | undefined>(undefined);

// Provider component
interface CurrConfigProviderProps {
  children: ReactNode;
}

export const CurrConfigProvider: React.FC<CurrConfigProviderProps> = ({ children }) => {
  // Load user from localStorage on initial render
  const [user, setUser] = useState<User | null>(() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  });

  // Store user in localStorage whenever it changes
  useEffect(() => {
    if (user) {
      localStorage.setItem("user", JSON.stringify(user));
    } else {
      localStorage.removeItem("user");
    }
  }, [user]);

  return (
    <CurrConfigContext.Provider value={{ user, setUser }}>
      {children}
    </CurrConfigContext.Provider>
  );
};
