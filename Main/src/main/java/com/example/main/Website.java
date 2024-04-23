package com.example.main;

import java.time.LocalDate;

public class Website
{
    private String URL;
    private String title;
    private LocalDate dateAccessed;

    public Website(String URL, String title)
    {
        this.URL = URL;
        this.title = title;
    }

    public Website(String URL, String title, LocalDate dateAccessed)
    {
        this.URL = URL;
        this.title = title;
        this.dateAccessed = dateAccessed;
    }

    public String getURL()
    {
        return URL;
    }

    public void setURL(String URL)
    {
        this.URL = URL;
    }

    public String getTitle()
    {
        return title;
    }

    public void setTitle(String title)
    {
        this.title = title;
    }

    public LocalDate getDateAccessed()
    {
        return dateAccessed;
    }

    public void setDateAccessed(LocalDate dateAccessed)
    {
        this.dateAccessed = dateAccessed;
    }
}
