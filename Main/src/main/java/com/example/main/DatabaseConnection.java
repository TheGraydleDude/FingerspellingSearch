package com.example.main;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class DatabaseConnection
{
    private String connectionUrl = "jdbc:sqlserver://AUSTEN_F\\SQLEXPRESS:1433;DatabaseName=ComputerScienceProject;integratedSecurity=true;trustServerCertificate=true;";

    private Connection connection = null;
    private Statement statement = null;
    private ResultSet resultSet = null;

    public void establishConnection()
    {
        try
        {
            Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver");
            connection = DriverManager.getConnection(connectionUrl);
        }
        catch (Exception e)
        {
            System.out.println("SQL Error 1: Unable to connect to database");
            e.printStackTrace();
        }
    }

    public ResultSet runQuery(String SQLQuery)
    {
        try
        {
            statement = connection.createStatement();
            resultSet = statement.executeQuery(SQLQuery);
        }
        catch (Exception e)
        {
            System.out.println("SQL Error 2: Invalid SQL Query");
            e.printStackTrace();
        }
        return resultSet;
    }

    public Boolean runUpdate(String insertQuery, int expectedNumOfChanges)
    {
        try
        {
            statement = connection.createStatement();
            int numOfChanges = statement.executeUpdate(insertQuery);
            if(numOfChanges == expectedNumOfChanges)
            {
                return true;
            }
            return false;
        }
        catch (Exception e)
        {
            System.out.println("SQL Error 4: Invalid Update Query");
            e.printStackTrace();
            return false;
        }
    }

    public void closeConnection()
    {
        if (connection != null)
        {
            try
            {
                connection.close();
            }
            catch (Exception e)
            {
                System.out.println("SQL Error 5: Cannot close connection");
                e.printStackTrace();
            }
        }

        if (statement != null)
        {
            try
            {
                statement.close();
            }
            catch (Exception e)
            {
                System.out.println("SQL Error 6: Cannot close statement");
                e.printStackTrace();
            }
        }

        if (resultSet != null)
        {
            try
            {
                resultSet.close();
            }
            catch (Exception e)
            {
                System.out.println("SQL Error 7: Cannot close result set");
                e.printStackTrace();
            }
        }
    }
}
