module com.example.main {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;
    requires java.sql;
    requires com.microsoft.sqlserver.jdbc;
    requires opencv;


    opens com.example.main to javafx.fxml;
    exports com.example.main;
}