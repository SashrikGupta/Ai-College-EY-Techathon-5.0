import React  , {useState , useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import LandingPage from './pages/LandingPage';
import AboutPage from './pages/AboutPage';
import LoginPage from './pages/LoginPage';
import ProfilePage from './pages/ProfilePage';
import AdvisorPage from './pages/AdvisorPage';
import LibrarianPage from './pages/LibrarianPage';
import IIICellPage from './pages/IIICellPage';
import FacultyPage from './pages/FacultyPage';
import ResearchPage from './pages/ResearchPage';
import ReportsPage from './pages/reports'; 
import Courses from './pages/courses'
import CourseForm from './pages/courseform'
import CourseList from './pages/courselist'
import ChatbotPage from './pages/ChatbotPage';
import InterviewPage from './pages/InterviewPage';
import { CurrConfigContext } from './context';
function App() {
  // In a real app, this would come from an auth context

  const cont = useContext(CurrConfigContext) ; 

  const isLoggedIn = cont.isLoggedIn ; 


  return (
    <Router>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <Navbar isLoggedIn={isLoggedIn} />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route 
            path="/login" 
            element={<LoginPage onLogin={()=>{
              cont.setIsLoggedIn(true) ;
            }} />} 
          />
          
          {/* Protected Routes */}
          <Route
            path="/new-course"
            element={isLoggedIn ? < CourseForm  /> : <Navigate to="/login" />}
          />
          <Route
          path="/profile"
          element={isLoggedIn ? < ProfilePage  /> : <Navigate to="/login" />}
          />
           <Route
          path="/interview"
          element={isLoggedIn ? < InterviewPage  /> : <Navigate to="/login" />}
          />
          <Route
            path="/previous-courses"
            element={isLoggedIn ? < CourseList /> : <Navigate to="/login" />}
          />
          <Route
            path="/courses"
            element={isLoggedIn ? <Courses /> : <Navigate to="/login" />}
          />
          <Route
            path="/courses"
            element={isLoggedIn ? <Courses /> : <Navigate to="/login" />}
          />
          <Route
            path="/reports"
            element={isLoggedIn ? <ReportsPage /> : <Navigate to="/login" />}
          />
          <Route
            path="/advisor"
            element={isLoggedIn ? <AdvisorPage /> : <Navigate to="/login" />}
          />
          <Route
            path="/librarian"
            element={isLoggedIn ? <LibrarianPage /> : <Navigate to="/login" />}
          />
          <Route
            path="/iii-cell"
            element={isLoggedIn ? <IIICellPage /> : <Navigate to="/login" />}
          />
          <Route
            path="/faculty"
            element={isLoggedIn ? <FacultyPage /> : <Navigate to="/login" />}
          />
          <Route
            path="/research"
            element={isLoggedIn ? <ResearchPage /> : <Navigate to="/login" />}
          />
          <Route
            path="/chat"
            element={isLoggedIn ? <ChatbotPage /> : <Navigate to="/login" />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;