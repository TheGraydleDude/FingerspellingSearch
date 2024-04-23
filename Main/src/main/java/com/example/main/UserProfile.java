package com.example.main;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

public class UserProfile
{
    private String username;
    private Queue<Website> history = new LinkedList<Website>() {};
    private ArrayList<Website> bookmarks = new ArrayList<>();

    public UserProfile()
    {
        this.username = "Guest";
    }


    public UserProfile(String username)
    {
        this.username = username;
    }

    public String getUsername()
    {
        return username;
    }

    public void setUsername(String username)
    {
        this.username = username;
    }

    public Queue<Website> getHistory()
    {
        while(this.history.size() > 10)
        {
            this.history.remove();
        }

        return history;
    }

    public void setHistory(Queue<Website> history) { this.history = history; }

    public ArrayList<Website> getBookmarks() {
        return bookmarks;
    }

    public void setBookmarks(ArrayList<Website> bookmarks) {
        this.bookmarks = bookmarks;
    }

    public void removeBookmark(String URL, String title)
    {
        for (int i = 0; i < bookmarks.size(); i++)
        {
            if(bookmarks.get(i).getURL().equals(URL) && bookmarks.get(i).getTitle().equals(title))
            {
                bookmarks.remove(i);
                return;
            }
        }
    }

    public Boolean containsBookmark(String URL, String title)
    {
        for (int i = 0; i < bookmarks.size(); i++)
        {
            if(bookmarks.get(i).getURL().equals(URL) && bookmarks.get(i).getTitle().equals(title))
            {
                return true;
            }
        }
        return false;
    }
}
