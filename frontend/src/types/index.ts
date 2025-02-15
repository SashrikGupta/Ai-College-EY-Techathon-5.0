export interface User {
  id: string;
  name: string;
  email: string;
  avatar: string;
  role: 'student' | 'faculty';
}

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'test' | 'job' | 'general';
  date: string;
  link : string
}

export interface Activity {
  id: string;
  type: 'test' | 'file' | 'chat';
  title: string;
  date: string;
  status?: 'completed' | 'pending';
}

export interface JobListing {
  id: string;
  title: string;
  company: string;
  location: string;
  type: string;
  applyLink: string;
}

export interface Course {
  id: string;
  title: string;
  platform: string;
  duration: string;
  level: string;
  link: string;
}