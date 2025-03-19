// CourseForm.tsx
import React, { useState, useEffect } from "react";
import {
  TextField,
  Button,
  Typography,
  Container,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import coursedata from "../courses.json";
import ChatbotPage from "./ChatbotPage"
export interface FormData {
  department: string;
  branch: string;
  interest: string;
  numberOfCourses: string;
  duration: string;
}

interface CourseFormProps {
  onSubmit: (data: FormData) => void;
}

const CourseForm: React.FC<CourseFormProps> = ({ onSubmit }) => {
  const [department, setDepartment] = useState("School of Engineering");
  const [branch, setBranch] = useState("");
  const [interest, setInterest] = useState("");
  const [numberOfCourses, setNumberOfCourses] = useState("");
  const [duration, setDuration] = useState("");
  const [branches, setBranches] = useState<string[]>([]);

  const departments = Object.keys(coursedata);

  useEffect(() => {
    // Update branches based on the selected department
    if (coursedata[department] && coursedata[department].departments) {
      setBranches(Object.keys(coursedata[department].departments));
    }
  }, [department]);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSubmit({ department, branch, interest, numberOfCourses, duration });
  };

  return (
    <Container maxWidth="sm">
      <Typography
        variant="h3"
        align="center"
        gutterBottom
        style={{ color: "#382D76", fontWeight: "bold" }}
      >
        Course Scheduler Portal
      </Typography>
      <Typography
        variant="body1"
        align="center"
        style={{ color: "#382D76" }}
      >
        Please enter your details below.
      </Typography>
      <Divider style={{ marginBottom: "2rem" }} />
{/* 
      <form
        onSubmit={handleSubmit}
        className="bg-white border border-gray-300 rounded-lg shadow-sm p-8 mx-auto"
      >
        <Typography
          variant="h5"
          align="center"
          gutterBottom
          style={{ color: "#382D76" }}
        >
          Course Timeline Input
        </Typography>

        <FormControl fullWidth margin="normal">
          <InputLabel>Department</InputLabel>
          <Select
            value={department}
            onChange={(e) => setDepartment(e.target.value)}
            label="Department"
          >
            {departments.map((depart, index) => (
              <MenuItem key={index} value={depart}>
                {depart}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl fullWidth margin="normal">
          <InputLabel>Branch</InputLabel>
          <Select
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            label="Branch"
          >
            {branches.map((br, index) => (
              <MenuItem key={index} value={br}>
                {br}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <TextField
          label="Enter Interest"
          variant="outlined"
          fullWidth
          value={interest}
          onChange={(e) => setInterest(e.target.value)}
          margin="normal"
        />

        <FormControl fullWidth margin="normal">
          <InputLabel>Number of Courses</InputLabel>
          <Select
            value={numberOfCourses}
            onChange={(e) => setNumberOfCourses(e.target.value)}
            label="Number of Courses"
          >
            {[1, 2, 3, 4, 5, 6, 7].map((num) => (
              <MenuItem key={num} value={num}>
                {num}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <TextField
          label="Duration (in days)"
          variant="outlined"
          fullWidth
          type="number"
          value={duration}
          onChange={(e) => {
            const value = e.target.value;
            if (/^\d*$/.test(value)) {
              setDuration(value);
            }
          }}
          margin="normal"
          inputProps={{ min: 1, max: 365, step: 1 }}
        />

        <Button
          variant="contained"
          type="submit"
          fullWidth
          size="large"
          style={{ backgroundColor: "#382D76", color: "#fff", marginTop: "1.5rem" }}
        >
          Submit
        </Button>
      </form> */}
      <ChatbotPage/>
    </Container>
  );
};

export default CourseForm;
