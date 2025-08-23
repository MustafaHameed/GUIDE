# SAM Dataset Event Log Schema

## Overview
This document defines the event log schema for the SAM (Student Action Mining) dataset used in process mining analysis.

## Required Fields

### case_id
- **Type**: String/Integer
- **Description**: Unique identifier for each student session or learning case
- **Example**: `student_12345`, `session_789`
- **Notes**: Groups events belonging to the same process instance

### activity  
- **Type**: String
- **Description**: Name of the action or learning activity performed
- **Example**: `login`, `view_video`, `submit_assignment`, `take_quiz`
- **Notes**: Should be descriptive and consistent across the dataset

### timestamp
- **Type**: DateTime (ISO 8601 format recommended)
- **Description**: When the activity occurred
- **Example**: `2023-09-15T10:30:00Z`, `2023-09-15 10:30:00`
- **Notes**: Must be sortable to establish temporal ordering

## Optional Fields

### resource
- **Type**: String  
- **Description**: Who or what performed the activity
- **Example**: `student`, `system`, `instructor`
- **Default**: `system` if not specified

### course
- **Type**: String
- **Description**: Course or module identifier
- **Example**: `MATH101`, `CS250`

### module  
- **Type**: String
- **Description**: Specific learning module or chapter
- **Example**: `Chapter_1`, `Linear_Algebra`

### device
- **Type**: String
- **Description**: Device type used for the activity
- **Example**: `desktop`, `mobile`, `tablet`

### success_flag
- **Type**: Boolean
- **Description**: Whether the activity was completed successfully
- **Example**: `true`, `false`

## Data Quality Requirements

1. **Completeness**: All required fields must be present
2. **Consistency**: Activity names should follow consistent naming conventions
3. **Temporal Ordering**: Timestamps must allow proper sequence reconstruction
4. **Case Grouping**: Each case_id should represent a coherent process instance

## Example Event Log Structure

```csv
case_id,activity,timestamp,resource,course,success_flag
student_001,login,2023-09-15T09:00:00Z,student,MATH101,true
student_001,view_material,2023-09-15T09:05:00Z,student,MATH101,true
student_001,take_quiz,2023-09-15T09:15:00Z,student,MATH101,false
student_001,logout,2023-09-15T09:30:00Z,student,MATH101,true
student_002,login,2023-09-15T10:00:00Z,student,CS250,true
```

## Process Mining Applications

This schema enables analysis of:
- Learning behavior patterns
- Activity sequences and variants
- Performance bottlenecks
- Conformance to expected learning paths
- Resource utilization patterns

## License

The SAM dataset schema is provided without an explicit license. Verify permissions with the data owner before distribution or reuse.