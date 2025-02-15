// CreateCourseForm.tsx
import React, { useState, useContext, useEffect } from "react";
import {
  Container,
  TextField,
  Button,
  Typography,
  Snackbar,
  Alert,
  MenuItem,
} from "@mui/material";
import { CurrConfigContext } from "../context.tsx";
import Stanford from "../courses.json";

interface FormData {
  userid: string;
  department: string;
  branch: string;
  interest: string;
  duration: string;
  n_course: number;
}

const CreateCourseForm: React.FC = () => {
  // Get the user from context
  const { user } = useContext(CurrConfigContext) || {};

  // Initialize form data using the userid from context (if available)
  const [formData, setFormData] = useState<FormData>({
    userid: user?._id || "",
    department: "School of Engineering", // default department
    branch: "",
    interest: "",
    duration: "",
    n_course: 1,
  });

  const [loading, setLoading] = useState(false);
  const [toastOpen, setToastOpen] = useState(false);
  const [error, setError] = useState("");
  // State to keep track of the selected department for branch options
  const [selectedDep, setSelectedDep] = useState("School of Engineering");

  // Update formData if the user context changes
  useEffect(() => {
    if (user?._id) {
      setFormData((prev) => ({ ...prev, userid: user._id }));
    }
  }, [user]);

  // Build department options from the keys in the imported JSON
  const departmentOptions = Object.keys(Stanford).map((department) => ({
    value: department,
    label: department,
  }));

  // Branch options depend on the selected department
  const branchOptions =
    selectedDep && Stanford[selectedDep] && Stanford[selectedDep].departments
      ? Object.keys(Stanford[selectedDep].departments).map((branch) => ({
          value: branch,
          label: branch,
        }))
      : [];

  // Handle input field changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    if (name === "department") {
      // Update both the selected department and form data,
      // and reset branch when department changes
      setSelectedDep(value);
      setFormData((prev) => ({
        ...prev,
        department: value,
        branch: "",
      }));
    } else {
      setFormData((prev) => ({
        ...prev,
        [name]: name === "n_course" ? Number(value) : value,
      }));
    }
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("http://127.0.0.1:5000/generate_course", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Error generating course");
      }
      // If successful, open the toast notification
      setToastOpen(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: "2rem" }}>
      <Typography
        variant="h4"
        gutterBottom
        style={{ color: "#382D76", textAlign: "center" }}
      >
        Create New Course
      </Typography>
      <form onSubmit={handleSubmit}>
        {/* Department Dropdown */}
        <TextField
          select
          label="Department"
          name="department"
          value={formData.department}
          onChange={handleChange}
          fullWidth
          margin="normal"
          required
        >
          {departmentOptions.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>

        {/* Branch Dropdown */}
        <TextField
          select
          label="Branch"
          name="branch"
          value={formData.branch}
          onChange={handleChange}
          fullWidth
          margin="normal"
          required
          disabled={!formData.department} // optionally disable if no department selected
        >
          {branchOptions.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>

        <TextField
          label="Interest"
          name="interest"
          value={formData.interest}
          onChange={handleChange}
          fullWidth
          margin="normal"
          required
        />
        <TextField
          label="Duration"
          name="duration"
          value={formData.duration}
          onChange={handleChange}
          fullWidth
          margin="normal"
          required
        />
        <TextField
          label="Number of Courses"
          name="n_course"
          type="number"
          value={formData.n_course}
          onChange={handleChange}
          fullWidth
          margin="normal"
          required
        />
        <Button
          type="submit"
          variant="contained"
          fullWidth
          style={{
            backgroundColor: "#382D76",
            color: "#fff",
            padding: "1rem",
            marginTop: "1rem",
          }}
        >
          Submit
        </Button>
      </form>
      {/* Show loader while waiting for the response */}
      {loading && <div className="loader" style={{ marginTop: "1rem" }} />}
      {/* Display error if any */}
      {error && (
        <Typography variant="body2" color="error" style={{ marginTop: "1rem" }}>
          {error}
        </Typography>
      )}
      {/* Snackbar toast */}
      <Snackbar
        open={toastOpen}
        autoHideDuration={6000}
        onClose={() => setToastOpen(false)}
      >
        <Alert
          onClose={() => setToastOpen(false)}
          severity="success"
          sx={{ width: "100%" }}
        >
          Your course has been made. Go to study section to see your course.
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default CreateCourseForm;
