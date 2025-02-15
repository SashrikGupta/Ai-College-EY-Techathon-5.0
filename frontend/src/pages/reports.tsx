// ReportsPage.tsx
import React, { useEffect, useState  , useContext} from 'react';
import { CurrConfigContext } from '../context.tsx';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title as ChartTitle,
  Tooltip,
  Legend,
} from 'chart.js';
import { Chart } from 'react-chartjs-2';

// Register required Chart.js modules
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ChartTitle,
  Tooltip,
  Legend
);

// Define TypeScript types for clarity (optional)
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

const ReportsPage: React.FC = () => {
   const cont = useContext(CurrConfigContext) || {};
  // State for reports fetched from the server
  const [reports, setReports] = useState<ReportData[]>([]);
  // State for the currently selected report index and carousel index
  const [selectedReportIndex, setSelectedReportIndex] = useState(0);
  const [carouselIndex, setCarouselIndex] = useState(0);

  // Fetch the reports from the server on component mount
  useEffect(() => {
    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    const raw = JSON.stringify({
      "userid": cont?.user._id
    });

    const requestOptions = {
      method: "POST",
      headers: myHeaders,
      body: raw,
      redirect: "follow" as RequestRedirect,
    };

    fetch("http://127.0.0.1:5000/test_reports", requestOptions)
      .then(response => response.json())
      .then((data: ReportData[]) => {
        setReports(data);
      })
      .catch((error) => console.error("Error fetching reports:", error));
  }, []);

  // Show a loading indicator if no reports have been loaded yet.
  if (reports.length === 0) {
    return <div className="flex items-center justify-center h-screen text-xl">
      <div className='loader'></div>
    </div>;
  }

  // Get the selected report and its questions array
  const selectedReport = reports[selectedReportIndex];
  const questions = selectedReport.report.report;

  // Calculate total marks for the current report
  const totalMarks = questions.reduce((sum, q) => sum + q.marks, 0);

  // Prepare chart labels and marks data
  const labels = questions.map((_, idx) => `Q${idx + 1}`);
  const marksData = questions.map((q) => q.marks);

  // Combined chart data (bar and line)
  const data = {
    labels,
    datasets: [
      {
        type: 'bar' as const,
        label: 'Marks',
        data: marksData,
        backgroundColor: 'rgba(136, 132, 216, 0.7)',
        borderColor: 'rgba(136, 132, 216, 1)',
        borderWidth: 1,
      },
      {
        type: 'line' as const,
        label: 'Trend',
        data: marksData,
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 2,
        fill: false,
        tension: 0.4,
      },
    ],
  };

  // Chart options with y-axis capped at 10
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 10,
        ticks: {
          stepSize: 1,
        },
      },
    },
    plugins: {
      legend: { position: 'top' as const },
      title: {
        display: true,
        text: 'Marks Overview',
      },
    },
  };

  // Format ISO timestamp to dd-mm-yyyy
  const formatDate = (timestamp: string): string => {
    const date = new Date(timestamp);
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const year = date.getFullYear();
    return `${day}-${month}-${year}`;
  };

  // When a report is clicked in the sidebar, update the selected report and reset the carousel index.
  const handleSelectReport = (index: number) => {
    setSelectedReportIndex(index);
    setCarouselIndex(0);
  };

  // Carousel navigation
  const handlePrev = () => {
    if (carouselIndex > 0) setCarouselIndex(carouselIndex - 1);
  };

  const handleNext = () => {
    if (carouselIndex < questions.length - 1) setCarouselIndex(carouselIndex + 1);
  };

  return (
    <div className="flex">
      {/* Sidebar with top and right margins */}
      <aside className="w-1/4 bg-white border-r p-4 mt-4 mr-4 overflow-auto h-[90vh] rounded-lg">
        <h2 className="text-xl font-bold mb-4">Reports</h2>
        <ul>
          {reports.map((report, idx) => (
            <li
              key={report._id}
              onClick={() => handleSelectReport(idx)}
              className={`cursor-pointer p-2 rounded mb-2 ${
                idx === selectedReportIndex
                  ? 'bg-primary-500 text-white'
                  : 'hover:bg-gray-200'
              }`}
            >
              {formatDate(report.time_stamp)}
            </li>
          ))}
        </ul>
      </aside>

      {/* Main content area: Right column split vertically */}
      <main className="flex-1 p-4 flex flex-col gap-4 overflow-auto h-[93vh]">
        {/* Top Panel: Display report title, total marks, and date */}
        <div className="h-[5vh] bg-white p-4 rounded shadow flex flex-col justify-center">
          <h2 className="text-xl font-bold text-center">
            {selectedReport.report.title} | Total Marks: {totalMarks} | {formatDate(selectedReport.time_stamp)}
          </h2>
        </div>

        {/* Carousel Panel: Showing question details with vertical scroll */}
        <div className="flex-1 bg-white p-4 rounded shadow flex flex-col h-[40vh] overflow-y-auto">
          <h3 className="text-lg font-bold mb-2">Question Details</h3>
          {questions.length > 0 && (
            <div className="flex-1">
              <div className="mb-2">
                <p className="text-sm font-medium text-gray-500">Title:</p>
                <p className="text-md">{questions[carouselIndex].title}</p>
              </div>
              <div className="mb-2">
                <p className="text-sm font-medium text-gray-500">Marks:</p>
                <p className="text-md">{questions[carouselIndex].marks}</p>
              </div>
              <div className="mb-2">
                <p className="text-sm font-medium text-gray-500">Question:</p>
                <p className="text-md font-bold">
                  {questions[carouselIndex].question}
                </p>
              </div>
              <div className="mb-2">
                <p className="text-sm font-medium text-gray-500">User Answer:</p>
                <p className="text-md">{questions[carouselIndex].answer}</p>
              </div>
              <div className="mb-2">
                <p className="text-sm font-medium text-gray-500">
                  AI Reason for Evaluation:
                </p>
                <p className="text-md">{questions[carouselIndex].reason}</p>
              </div>
            </div>
          )}
          <div className="flex justify-between mt-4">
            <button
              onClick={handlePrev}
              disabled={carouselIndex === 0}
              className="px-4 py-2 bg-primary-500 text-white rounded disabled:opacity-50"
            >
              Prev
            </button>
            <button
              onClick={handleNext}
              disabled={carouselIndex === questions.length - 1}
              className="px-4 py-2 bg-primary-500 text-white rounded disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>

        {/* Bottom Panel: Combined Chart */}
        <div className="h-[35vh] bg-white p-4 rounded shadow">
          <Chart type="bar" data={data} options={options} />
        </div>
      </main>
    </div>
  );
};

export default ReportsPage;
