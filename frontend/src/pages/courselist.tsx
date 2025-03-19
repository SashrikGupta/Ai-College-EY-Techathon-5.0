import React, { useState, useEffect, useContext } from 'react';
import CourseInfo from './Courseinfo';
import TopicInfo from './topicinfo';
import { CurrConfigContext } from '../context.tsx';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';

const CourseList = () => {
  const cont = useContext(CurrConfigContext) || {};
  const [courses, setCourses] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [loading, setLoading] = useState(true);
  // activeView controls what is shown in the main content area.
  // "info" means show <CourseInfo>, any other value means show "Select a course"
  const [activeView, setActiveView] = useState('none');

  const[selectednodeindex , setselectednodeindex] = useState(0)

  const[selectedsubnodeindex , setselectedsubnodeindex] = useState(-1)
  const[selectedtopicindex , setselectedtopicindex] = useState(-1)

  const[infoview , setinfoview] = useState(true)

  // Fetch courses when the component mounts
  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const response = await fetch('http://localhost:5000/get_courses', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ userid: cont?.user?._id }),
        });
        const data = await response.json();
        console.log(data[0]);
        setCourses(data);
        // Set the first course as selected by default
        if (data.length > 0) {
          setSelectedCourse(data[0]);
        }
        setLoading(false);
      } catch (error) {
        console.error('Error fetching courses:', error);
        setLoading(false);
      }
    };

    fetchCourses();
  }, [cont?.user?._id]);

  useEffect(() => {
    console.log(infoview)
  }, [infoview]);

  // Handle node selection from the tree view
  const handleNodeSelect = (e) => {
    const itemId = e.node.itemId;
    if (itemId.endsWith('-info')) {
      // "Info" node clicked: extract the course id and update the state.
      const courseId = itemId.replace('-info', '');
      const selected = courses.find((course) => course._id === courseId);
      if (selected) {
        setSelectedCourse(selected);
        setActiveView('info');
      }
    } else if (e.node.data && e.node.data.type === 'course') {
      // Course node clicked: update the selected course but do not show info.
      const selected = courses.find((course) => course._id === e.node.itemId);
      if (selected) {
        setSelectedCourse(selected);
        setActiveView('none');
      }
    } else {
      // For any other node, reset the view.
      setActiveView('none');
    }
  };

  if (loading) {
    return (
      <div className="w-[97vw] h-[95vh] flex items-center justify-center">
        <div className="loader"></div>
      </div>
    );
  }



  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar using Tree view */}
      <aside
        style={{
          width: '250px',
          borderRight: '1px solid #ccc',
          padding: '20px',
          boxSizing: 'border-box',
          overflowY: 'auto',
        }}
      >
        <SimpleTreeView>
          {courses.map((course , idx) => {
            return (
              <TreeItem
                key={course._id}
                itemId={course._id}
                label={course.name}
                onSelect={handleNodeSelect}
                data={{ type: 'course' }}
                onClick={()=>{
                  setselectednodeindex(idx)
                  setinfoview(true)

               }}
              >

                {course.course.content.map((subCourse, subIndex) => {
                  return (
                    <TreeItem
                      key={`${course._id}-sub-${subIndex}`}
                      itemId={`${course._id}-sub-${subIndex}`}
                      label={subCourse.name || `Sub Course ${subIndex + 1}`}
                      onSelect={handleNodeSelect}
                    >
                      {subCourse.timeline.map((topic, topicIndex) => {
                        if (topic.activity_type === 1) {
                          return (
                            <TreeItem
                              key={`${course._id}-sub-${subIndex}-topic-${topicIndex}`}
                              itemId={`${course._id}-sub-${subIndex}-topic-${topicIndex}`}
                              label={topic.topic_name}
                              onClick={()=>{
                                setselectedsubnodeindex(subIndex)
                                setselectedtopicindex(topicIndex)
                                setinfoview(false)                                
                              }}
                            />
                          );
                        }
                        return null;
                      })}
                    </TreeItem>
                  );
                })}
              </TreeItem>
            );
          })}
        </SimpleTreeView>
      </aside>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
      
        { infoview ? <CourseInfo newCoursesData={courses[selectednodeindex]} />
        :
        <div>
        <TopicInfo topic = {courses[selectednodeindex].course.content[selectedsubnodeindex].timeline[selectedtopicindex]} subcourseidx = {selectedsubnodeindex} topicidx = {selectedtopicindex} courseid = {courses[selectednodeindex]._id}/>
       </div>
        
        }
      </main>
    </div>
  );
};

export default CourseList;
