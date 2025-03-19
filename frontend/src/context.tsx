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
  setIsLoggedIn: (isLoggedIn: boolean) => void;
  handleLogin: () => void;
  isLoggedIn: boolean;
}

// Create the context
export const CurrConfigContext = createContext<CurrConfigContextType | undefined>(undefined);

// Provider component
interface CurrConfigProviderProps {
  children: ReactNode;
}

export const CurrConfigProvider: React.FC<CurrConfigProviderProps> = ({ children }) => {
  // Load user and isLoggedIn state from localStorage on initial render
  const [user, setUser] = useState<User | null>(() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  });

  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(() => {
    const storedIsLoggedIn = localStorage.getItem("isLoggedIn");
    return storedIsLoggedIn ? JSON.parse(storedIsLoggedIn) : false;
  });

  // Store user in localStorage whenever it changes
  useEffect(() => {
    if (user) {
      localStorage.setItem("user", JSON.stringify(user));
      setIsLoggedIn(true);
    } else {
      localStorage.removeItem("user");
      setIsLoggedIn(false);
    }
  }, [user]);

  // Store isLoggedIn in localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("isLoggedIn", JSON.stringify(isLoggedIn));
    console.log(user?._id)
  }, [isLoggedIn]);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  return (
    <CurrConfigContext.Provider value={{ user, setUser, setIsLoggedIn, handleLogin, isLoggedIn }}>
      {children}
    </CurrConfigContext.Provider>
  );
};
