import React, { useContext, useEffect, useState } from 'react';
import CalendarHeatmap from 'react-calendar-heatmap';
import { Tooltip } from 'react-tooltip';
import { Bell, Calendar as CalendarIcon, Activity as ActivityIcon, Download, ChevronRight } from 'lucide-react';
import type { Notification, Activity } from '../types';
import { CurrConfigContext } from '../context.tsx';
import NewsAPI from 'newsapi';
import { Line } from "react-chartjs-2";
import { useNavigate } from "react-router-dom";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Legend,
} from "chart.js";
import ReactCalendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Legend);

interface Article {
  title: string;
  description: string;
  url: string;
  author: string;
  publishedAt?: string;
  content?: string;
  source?: { name?: string };
}

interface NewsApiResponse {
  status: string;
  totalResults: number;
  articles: Article[];
}

interface Question {
  answer: string;
  marks: number;
  question: string;
  reason: string;
  title: string;
}

interface ReportData {
  report: {
    report: Question[];
    review: string;
    title: string;
  };
  _id: string;
  student_id: string;
  time_stamp: string;
}

const apiKey = '9be285863c0548cc9108c65cd6fab48b';

const ProfilePage = () => {
  const navigate = useNavigate();
  const [reports, setReports] = useState<ReportData[]>([]);
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const API_URL = `https://newsapi.org/v2/everything?q=coding&sortBy=popularity&apiKey=${apiKey}`;

  // Fetch articles
  useEffect(() => {
    fetch(API_URL)
      .then((response) => response.json())
      .then((data: NewsApiResponse) => {
        if (data.articles) {
          const wiredArticles = data.articles.filter((art) => art.source?.name === 'Wired');
          setArticles(wiredArticles);
        } else {
          setError("No articles found.");
        }
        setLoading(false);
      })
      .catch((error) => {
        setError("Failed to fetch news. Please try again.");
        setLoading(false);
      });
  }, [API_URL]);

  // Utility function to format dates
  const give_format_date = (timestamp: string) => {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      console.error('Invalid date:', timestamp);
      return 'Invalid Date';
    }
    return `${date.getFullYear()}-${(date.getMonth() + 1)
      .toString()
      .padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')}`;
  };

  const [test_labels, set_test_labels] = useState<string[]>([]);
  const [test_values, set_test_values] = useState<number[]>([]);

  // Context & user state management
  const context = useContext(CurrConfigContext) as any; // adjust ContextType as needed
  const [cont, setCont] = useState(useContext(CurrConfigContext));

  // Fetch user and test reports
  useEffect(() => {
    const fetchUserAndReports = async () => {
      try {
        // Fetch user data if email exists
        if (cont.user?.email) {
          const response = await fetch('http://localhost:5000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: cont.user.email }),
          });
          if (!response.ok) throw new Error('Failed to fetch user data');
          const userData = await response.json();
          setCont({ user: userData });
        }

        // Fetch reports if user id exists
        if (cont.user?._id) {
          const myHeaders = new Headers();
          myHeaders.append("Content-Type", "application/json");

          const raw = JSON.stringify({ userid: cont.user._id });

          const requestOptions = {
            method: "POST",
            headers: myHeaders,
            body: raw,
            redirect: "follow" as RequestRedirect,
          };

          const reportsResponse = await fetch("http://127.0.0.1:5000/test_reports", requestOptions);
          const reportsData: ReportData[] = await reportsResponse.json();
          setReports(reportsData);
        }
      } catch (error) {
        console.error('Error fetching user or reports:', error);
      }
    };

    fetchUserAndReports();
  }, []); // re-run if cont.user changes

  // Update test_labels and test_values whenever reports change
  useEffect(() => {
    if (reports.length === 0) return;

    // Create labels by converting timestamp strings to readable dates
    const tempLabels = reports
      .map((report) => new Date(report.time_stamp).toLocaleString())
      .reverse();

    // Calculate total marks for each report
    const tempValues = reports
      .map((report) => {
        const questions = report.report.report;
        return questions.reduce((sum, q) => sum + q.marks, 0);
      })
      .reverse();

    set_test_labels(tempLabels);
    set_test_values(tempValues);
  }, [reports]);

  // Prepare data for the chart
  const data = {
    labels: test_labels, 
    datasets: [
      {
        label: "Test Performance",
        data: test_values,
        borderColor: "purple",
        backgroundColor: "rgba(128, 0, 128, 0.2)",
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 5,
        pointBackgroundColor: "purple",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: "top",
      },
      tooltip: {
        enabled: true, // Ensures tooltips show on hover
      },
    },
    scales: {
      x: {
        ticks: {
          display: false, // Hides x-axis labels
        },
      },
    },
  };

  // Prepare user, notifications, and activities for display
  const user = {
    name: cont.user?.name || 'User Name',
    email: cont.user?.email || 'user@example.com',
    avatar: cont?.user?.image_url || 'https://avatar.iran.liara.run/public/6',
  };

  const notifications = [
    {
      id: '1',
      title: articles[0]?.author,
      message: articles[0]?.content?.slice(0, 100) + "...",
      type: 'test',
      date: articles[0]?.publishedAt?.slice(0, 10),
      link: articles[0]?.url,
    },
    {
      id: '2',
      title: articles[1]?.author,
      message: articles[1]?.content?.slice(0, 100) + "...",
      type: 'test',
      date: articles[1]?.publishedAt?.slice(0, 10),
      link: articles[1]?.url,
    },
    {
      id: '3',
      title: articles[2]?.author,
      message: articles[2]?.content?.slice(0, 100) + "...",
      type: 'test',
      date: articles[2]?.publishedAt?.slice(0, 10),
      link: articles[2]?.url,
    },
  ];

  // Sort activities if available
  const sortedActivities = [...(cont.user?.activity || [])].sort(
    (a, b) => new Date(b.time_stamp).getTime() - new Date(a.time_stamp).getTime()
  );

  const activities: Activity[] = sortedActivities.slice(0, 3).map((activity, index) => ({
    id: (index + 1).toString(),
    type: activity.activity_name?.name || 'test',
    title: activity.activity_name?.name || 'Unnamed Activity',
    date: give_format_date(activity.time_stamp),
    status: 'completed',
  }));

  // Generate heatmap data (if activity exists)
  const today = new Date();
  const activityCountMap = (cont.user?.activity || []).reduce((map: Map<string, number>, act: any) => {
    const date = act.time_stamp.split('T')[0];
    map.set(date, (map.get(date) || 0) + 1);
    return map;
  }, new Map<string, number>());

  const activityData = [...Array(365)].map((_, i) => {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    const formattedDate = date.toISOString().split('T')[0];
    return {
      date: formattedDate,
      count: activityCountMap.get(formattedDate) || 0,
    };
  });
  activityData.reverse();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <img
              className="h-16 w-16 rounded-full"
              src={user.avatar}
              alt={user.name}
            />
            <div className="ml-4">
              <h2 className="text-xl font-bold text-gray-900">{user.name}</h2>
              <p className="text-sm text-gray-500">{user.email}</p>
            </div>
          </div>
          <button className="mt-4 w-full px-4 py-2 border border-primary-500 text-primary-500 rounded-md hover:bg-primary-50 transition-colors"
                onClick={() => {
                  context.setIsLoggedIn(false)
                  navigate("/"); // Navigate to home
                  console.log("hello")
                }}
          >
            Log out 
          </button>
        </div>

        {/* Activity Heatmap */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Activity Overview</h3>
          <div className="overflow-hidden">
            <CalendarHeatmap
              startDate={new Date(today.getFullYear(), today.getMonth() - 11, 1)}
              endDate={today}
              values={activityData}
              classForValue={(value) => {
                if (!value || value.count === 0) return 'color-empty';
                return `color-scale-${value.count}`;
              }}
              tooltipDataAttrs={(value: any) => {
                if (!value || !value.date) return null;
                return {
                  'data-tooltip-id': 'activity-tooltip',
                  'data-tooltip-content': `${value.date}: ${value.count} activities`,
                };
              }}
            />
            <Tooltip id="activity-tooltip" />
          </div>
        </div>

        {/* Recent Activities */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Recent Activities</h3>
            <ActivityIcon className="h-5 w-5 text-gray-400" />
          </div>
          <div className="space-y-4">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full ${
                    activity.status === 'completed' ? 'bg-green-500' : 'bg-yellow-500'
                  }`} />
                  <div className="ml-3">
                    <p className="font-medium text-gray-900">{activity.title}</p>
                    <p className="text-sm text-gray-500">{activity.date}</p>
                  </div>
                </div>
                <ChevronRight className="h-5 w-5 text-gray-400" />
              </div>
            ))}
          </div>
        </div>

        {/* Test Performance Chart */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6 h-[35vh]">
          <Line data={data} options={options} />
        </div>

        {/* Calendar */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Calendar</h3>
            <CalendarIcon className="h-5 w-5 text-gray-400" />
          </div>
          <ReactCalendar
            style={{ border: 'none', boxShadow: 'none' }}
            className="w-full"
          />

        </div>
            
        {/* Notifications */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6 ">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
            <Bell className="h-5 w-5 text-gray-400" />
          </div>
          <div className="space-y-4">
            {notifications.map((notification) => (
              <div
                key={notification.id}
                className="flex items-start p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex-1">
                  <a className="font-medium text-gray-900" href={notification.link}>{notification.title}</a>
                  <p className="text-sm text-gray-500">{notification.message}</p>
                  <p className="text-xs text-gray-400 mt-1">{notification.date}</p>
                </div>
                <ChevronRight className="h-5 w-5 text-gray-400" href={notification.link} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
