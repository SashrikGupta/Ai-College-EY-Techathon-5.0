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
  Grid,
  Slide,
} from "@mui/material";
import { CurrConfigContext } from "../context.tsx";
import Stanford from "../courses.json";
import ChatbotPage from "./ChatbotPage";

interface FormData {
  userid: string;
  department: string;
  branch: string;
  interest: string;
  duration: string;
  n_course: number;
  index_course: string; // stored as a string from the input
  remarks: string; // added for advisor remarks
}

const CreateCourseForm: React.FC = () => {
  const { user } = useContext(CurrConfigContext) || {};

  const [formData, setFormData] = useState<FormData>({
    userid: user?._id || "",
    department: "School of Engineering",
    branch: "",
    interest: "",
    duration: "",
    n_course: 1,
    index_course: "",
    remarks: "", // initial state for remarks
  });
  const [loading, setLoading] = useState(false);
  const [toastOpen, setToastOpen] = useState(false);
  const [error, setError] = useState("");
  const [selectedDep, setSelectedDep] = useState("School of Engineering");
  const [showChatbot, setShowChatbot] = useState(false);

  useEffect(() => {
    if (user?._id) {
      setFormData((prev) => ({ ...prev, userid: user._id }));
    }
  }, [user]);

  const departmentOptions = Object.keys(Stanford).map((department) => ({
    value: department,
    label: department,
  }));

  const branchOptions =
    selectedDep && Stanford[selectedDep] && Stanford[selectedDep].departments
      ? Object.keys(Stanford[selectedDep].departments).map((branch) => ({
          value: branch,
          label: branch,
        }))
      : [];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    if (name === "department") {
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

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    // Create payload with index_course as a list
    const payload = formData;

    try {
      const response = await fetch("http://127.0.0.1:5000/generate_course", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Error generating course");
      }
      setToastOpen(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const CourseFormCard = (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <Typography
        variant="h5"
        gutterBottom
        style={{ color: "#382D76", textAlign: "center" }}
      >
        Create New Course
      </Typography>
      <form onSubmit={handleSubmit}>
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
        <TextField
          select
          label="Branch"
          name="branch"
          value={formData.branch}
          onChange={handleChange}
          fullWidth
          margin="normal"
          required
          disabled={!formData.department}
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
          label="Enter the number of courses"
          name="n_course"
          value={formData.n_course}
          onChange={handleChange}
          fullWidth
          type="number"
          margin="normal"
          required
        />
        {/* New TextField for advisor remarks */}
        <TextField
          label="Remarks given by advisor"
          name="remarks"
          value={formData.remarks}
          onChange={handleChange}
          fullWidth
          margin="normal"
        />
        {/* Chatbot can be initialized with department, branch, and interest only */}
        {!showChatbot && (
          <Button
            variant="outlined"
            fullWidth
            style={{
              padding: "1rem",
              marginTop: "1rem",
              marginBottom: "5rem",
            }}
            onClick={() => setShowChatbot(true)}
            disabled={!formData.department || !formData.branch || !formData.interest}
          >
            Initialize Chat Bot
          </Button>
        )}
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
      {loading && <div className="loader" style={{ marginTop: "1rem" }} />}
      {error && (
        <Typography variant="body2" color="error" style={{ marginTop: "1rem" }}>
          {error}
        </Typography>
      )}
    </div>
  );

  return (
    <Container maxWidth="xl" style={{ marginTop: "2rem" }}>
      <Typography
        variant="h4"
        gutterBottom
        style={{ color: "#382D76", textAlign: "center" }}
      >
        Course Creation Assistant
      </Typography>
      <Grid container spacing={4} justifyContent="center">
        <Grid item xs={12} md={6}>
          <div
            style={{
              transition: "transform 0.5s",
              transform: showChatbot ? "translateX(-100px)" : "translateX(0)",
            }}
          >
            {CourseFormCard}
          </div>
        </Grid>
        {showChatbot && (
          <Grid item xs={12} md={6}>
            <Slide direction="left" in={showChatbot} mountOnEnter unmountOnExit>
              <div>
                <ChatbotPage
                  department={formData.department}
                  branch={formData.branch}
                  interest={formData.interest}
                />
              </div>
            </Slide>
          </Grid>
        )}
      </Grid>
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
