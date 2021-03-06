/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

#ifndef TASKLIST_H
#define TASKLIST_H

#include<ostream>
#include<vector>
#include<string>

class TaskList
{

public:
  static void addTask(std::string task) {task_.push_back(task);}

  static void viewAllTasks(bool useFile = false);

private:
  TaskList();

  static std::vector<std::string> task_;

};

#endif
