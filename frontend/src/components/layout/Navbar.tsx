import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  NotebookText,  // Changed from BookOpen to NotebookText
  GraduationCap, 
  Users, 
  BrainCircuit, 
  FileText,
  User,
  Menu,
  BarChartBig,  
  BookOpen , 
  X , 
  LibraryBig , 
  BookCopy
} from 'lucide-react';

const Navbar = ({ isLoggedIn }: { isLoggedIn: boolean }) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const location = useLocation();

  const navigation = isLoggedIn ? [
    { name: 'Home', href: '/', icon: Home },,

    { name: 'Study' , href:'/previous-courses' , icon: LibraryBig}, // Updated icon
    { name: 'Course Bot ' , href:'/courses' , icon : BookCopy },
    { name: 'Faculty', href: '/faculty', icon: Users },
    { name: 'Reports', href: '/reports', icon: BarChartBig },
    { name: 'AI Advisor', href: '/advisor', icon: BrainCircuit } ,
    { name: 'III Cell', href: '/iii-cell', icon: GraduationCap },
    { name: 'Librarian', href: '/librarian', icon: NotebookText }, // Optionally change this too
    { name: 'ResearchGen', href: '/research', icon: FileText },
    { name: 'Profile', href: '/profile', icon: User },
  ] : [
    { name: 'Home', href: '/', icon: Home },
    { name: 'About', href: '/about', icon: NotebookText }, // Optionally change for consistency
  ];

  return (
    <nav className="bg-white shadow-lg dark:bg-gray-900 ">
      <div className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link to="/" className="flex-shrink-0 flex items-center">
              <BrainCircuit className="h-8 w-8 text-primary-500" />
              <span className="ml-2 text-xl font-bold text-primary-500">AI College</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:space-x-4">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`${
                    location.pathname === item.href
                      ? 'text-primary-500'
                      : 'text-gray-600 hover:text-primary-500'
                  } px-3 py-2 rounded-md text-sm font-medium flex items-center`}
                >
                  <Icon className="h-4 w-4 mr-1" />
                  {item.name}
                </Link>
              );
            })}
            {!isLoggedIn && (
              <Link
                to="/login"
                className="bg-primary-500 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-primary-600"
              >
                Login
              </Link>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="flex items-center md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-primary-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500"
            >
              {isOpen ? (
                <X className="block h-6 w-6" />
              ) : (
                <Menu className="block h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`${
                    location.pathname === item.href
                      ? 'text-primary-500'
                      : 'text-gray-600 hover:text-primary-500'
                  } block px-3 py-2 rounded-md text-base font-medium flex items-center`}
                  onClick={() => setIsOpen(false)}
                >
                  <Icon className="h-5 w-5 mr-2" />
                  {item.name}
                </Link>
              );
            })}
            {!isLoggedIn && (
              <Link
                to="/login"
                className="block w-full text-center bg-primary-500 text-white px-4 py-2 rounded-md text-base font-medium hover:bg-primary-600"
                onClick={() => setIsOpen(false)}
              >
                Login
              </Link>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
