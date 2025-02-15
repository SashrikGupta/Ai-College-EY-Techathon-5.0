// CourseDetails.tsx
import React, { useContext } from "react";
import { jsPDF } from "jspdf";
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Container,
  Divider,
  Button,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import AssignmentIcon from "@mui/icons-material/Assignment";
import SearchIcon from "@mui/icons-material/Search";
import { CurrConfigContext } from "../context.tsx";

// New courses data (using your provided JSON format)


// Helper functions to get the icon and activity label based on activity type
const getIcon = (activityType: number) => {
  switch (activityType) {
    case 1:
      return <MenuBookIcon style={{ color: "#382D76" }} />;
    case 2:
      return <AssignmentIcon style={{ color: "#382D76" }} />;
    case 3:
      return <SearchIcon style={{ color: "#382D76" }} />;
    default:
      return null;
  }
};

const getActivityLabel = (activityType: number) => {
  switch (activityType) {
    case 1:
      return "Library";
    case 2:
      return "Test";
    case 3:
      return "Research";
    default:
      return "";
  }
};

interface CourseDetailsProps {
  interest?: string;
}

const CourseDetails: React.FC<CourseDetailsProps> = (props) => {
  // Get the username from context
  console.log(props)
  const newCoursesData = props.newCoursesData
  console.log(newCoursesData.course.content)
  const cont = useContext(CurrConfigContext) || {};
  const username = cont?.user?.name || "User";
  const courses = newCoursesData?.course.content;

  

  // Function to generate a PDF with a cover page and course details
  const handleDownloadPdf = async () => {
    const doc = new jsPDF();

    // Utility function to load an image and convert it to a Data URL
    const getDataUrl = (url: string): Promise<string> => {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(img, 0, 0);
            const dataURL = canvas.toDataURL("image/png");
            resolve(dataURL);
          } else {
            reject(new Error("Could not get canvas context."));
          }
        };
        img.onerror = (err) => {
          reject(err);
        };
        img.src = url;
      });
    };

    try {
      const logoUrl =
        "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcQ86W-BPe7pHMN0NC7XzOLgsRXBJwJRFo73cjCGM8PHGzUgBvrY";
      const logoDataUrl = await getDataUrl(logoUrl);

      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();

      // COVER PAGE
      const logoWidth = 50;
      const logoHeight = 50;
      const logoX = (pageWidth - logoWidth) / 2;
      const logoY = pageHeight / 2 - logoHeight - 20;
      doc.addImage(logoDataUrl, "PNG", logoX, logoY, logoWidth, logoHeight);

      doc.setFontSize(16);
      doc.setTextColor(56, 45, 118);
      doc.text(
        "AI college Study Material",
        pageWidth / 2,
        logoY + logoHeight + 20,
        { align: "center" }
      );

      const titleText = `${newCoursesData?.name} course`;
      doc.setFontSize(20);
      doc.setTextColor(0);
      doc.text(titleText, pageWidth / 2, logoY + logoHeight + 35, {
        align: "center",
      });

      doc.setFontSize(16);
      doc.setTextColor(0);
      doc.text(
        `Personalised for ${username}`,
        pageWidth / 2,
        logoY + logoHeight + 45,
        { align: "center" }
      );

      doc.addPage();

      courses.forEach((course, courseIndex) => {
        if (courseIndex > 0) {
          doc.addPage();
        }

        doc.setTextColor(56, 45, 118);
        doc.setFontSize(20);
        doc.text(course.name, 20, 20);

        doc.setFontSize(12);
        doc.setTextColor(0);
        const descriptionLines = doc.splitTextToSize(course.description, 170);
        doc.text(descriptionLines, 20, 30);

        let y = 40 + descriptionLines.length * 7;

        course.timeline.forEach((entry) => {
          console.log(entry)
          doc.setFontSize(12);
          doc.setTextColor(56, 45, 118);
          const headerText = `Day ${entry.start_time} - Day ${entry.end_time} | ${getActivityLabel(
            entry.activity_type
          )}`;
          doc.text(headerText, 20, y);
          y += 7;

          doc.setFontSize(10);
          doc.setTextColor(0);
          const topicsText = `Topics: ${entry.topic_name}`;
          doc.text(topicsText, 25, y);
          y += 7;

          const entryDescLines = doc.splitTextToSize(entry.description, 160);
          doc.text(entryDescLines, 25, y);
          y += entryDescLines.length * 7;

          if (entry.study_material) {
            doc.setFontSize(11);
            doc.setTextColor(56, 45, 118);
            doc.text("Study Material:", 25, y);
            y += 7;
            entry.study_material.forEach((material) => {
              doc.setFontSize(10);
              doc.setTextColor(56, 45, 118);
              doc.text("Video Resources:", 30, y);
              y += 7;
              doc.setTextColor(0);
              material.video.forEach((videoItem) => {
                const videoText = `- ${videoItem.title}: ${videoItem.link}`;
                const videoLines = doc.splitTextToSize(videoText, 150);
                doc.text(videoLines, 35, y);
                y += videoLines.length * 7;
              });
              doc.setFontSize(10);
              doc.setTextColor(56, 45, 118);
              doc.text("Textual resources:", 30, y);
              y += 7;
              doc.setTextColor(0);
              material.pdf.forEach((googleItem) => {
                const googleText = `- ${googleItem.link}`;
                const googleLines = doc.splitTextToSize(googleText, 150);
                doc.text(googleLines, 35, y);
                y += googleLines.length * 7;
              });
            });
          }
          y += 5;
          if (y > 270) {
            doc.addPage();
            y = 20;
          }
        });
      });
      doc.save("courses.pdf");
    } catch (error) {
      console.error("Error loading logo image:", error);
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: "2rem" }}>
      <Typography
        variant="h5"
        align="center"
        gutterBottom
        style={{ color: "#382D76" }}
      >
        Course Details
      </Typography>
      <Typography
        variant="body2"
        align="center"
        style={{ color: "#382D76" , marginBottom: "1rem" , }}
      >
        Expand each course to view a detailed description, schedule, and study materials.
      </Typography>
      <Button
        variant="contained"
        fullWidth
        size="large"
        onClick={handleDownloadPdf}
        style={{
         marginBottom: "1rem" ,
          backgroundColor: "#382D76",
          color: "#fff",
          marginBottom: "1.5rem",
        }}
      >
        Download PDF
      </Button>
      {courses.map((course) => (
        <Accordion
          key={course.index}
          style={{
            marginBottom: "1rem" ,
            border: "1px solid #382D76",
            borderRadius: "4px",
            boxShadow: "0px 2px 4px rgba(0,0,0,0.1)",
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon style={{ color: "#382D76" }} />}
          >
            <Typography variant="h6" style={{ color: "#382D76" }}>
              {course.name}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body1" paragraph style={{ color: "#382D76" }}>
              {course.description}
            </Typography>
            <List>{
              console.log(course.timeline)
              }
              {course.timeline.map((entry, idx) => (
                <div key={idx}>
                  <ListItem alignItems="flex-start">
                    <ListItemIcon>{getIcon(entry.activity_type)}</ListItemIcon>
                    <ListItemText
                      primary={
                        <>
                          <Typography
                            variant="subtitle1"
                            style={{ color: "#382D76", fontWeight: "bold" }}
                          >
                            {`Day ${entry.start_time} - Day ${entry.end_time} | ${getActivityLabel(
                              entry.activity_type
                            )}`}
                          </Typography>
                          <Typography
                            variant="body2"
                            style={{
                              color: "#382D76",
                              fontStyle: "italic",
                            }}
                          >
                            {`Topics: ${entry.topic_name}`}
                          </Typography>
                        </>
                      }
                      secondary={
                        <Typography variant="body2" style={{ color: "#555" }}>
                          {entry.description}
                        </Typography>
                      }
                    />
                  </ListItem>
                  {entry.study_material && (
                    <Accordion
                      style={{
                        marginLeft: "40px",
                        marginBottom: "16px",
                        backgroundColor: "#f7f7f7",
                      }}
                    >
                      <AccordionSummary
                        expandIcon={<ExpandMoreIcon style={{ color: "#382D76" }} />}
                        aria-controls="study-material-content"
                        id="study-material-header"
                      >
                        <Typography
                          variant="subtitle2"
                          style={{ color: "#382D76" }}
                        >
                          Study Material
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        {console.log(entry.study_material)}
                        {entry.study_material.map((material, mIndex) => (
                          <div key={mIndex}>
                            <Typography
                              variant="subtitle2"
                              style={{
                                color: "#382D76",
                                marginLeft: "16px",
                                marginBottom: "4px",
                              }}
                            >
                              Video Resources:
                            </Typography>
                            <List dense>
                              {material.video.map((videoItem, vidIndex) => (
                                <ListItem key={`video-${vidIndex}`}>
                                  <ListItemText
                                    primary={videoItem.title}
                                    secondary={
                                      <a
                                        href={videoItem.link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                      >
                                        {videoItem.link}
                                      </a>
                                    }
                                  />
                                </ListItem>
                              ))}
                            </List>
                            <Typography
                              variant="subtitle2"
                              style={{
                                color: "#382D76",
                                marginLeft: "16px",
                                marginBottom: "4px",
                              }}
                            >
                              Google Resources:
                            </Typography>
                            <List dense>
                              {material.pdf.map((googleItem, gIndex) => (
                                <ListItem key={`google-${gIndex}`}>
                                  <ListItemText
                                    primary={
                                      <a
                                        href={googleItem.link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                      >
                                        {googleItem.link}
                                      </a>
                                    }
                                  />
                                </ListItem>
                              ))}
                            </List>
                            <Divider />
                          </div>
                        ))}
                      </AccordionDetails>
                    </Accordion>
                  )}
                  <Divider />
                </div>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      ))}
    </Container>
  );
};

export default CourseDetails;
