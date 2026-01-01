Web Interface Features
======================

Overview of the Pantheon web UI at https://pantheon-ui.vercel.app/.

Chat Interface
--------------

**Message Input**

- Type messages in the input box at the bottom
- Press Enter to send
- Use Shift+Enter for multi-line messages

**Message Display**

- User messages on the right
- Agent responses on the left
- Code blocks with syntax highlighting
- Markdown rendering for formatted text

**Agent Indicators**

- Shows which agent is responding (in team mode)
- Displays agent icon and name

Session Management
------------------

**New Session**

Click "New Chat" to start a fresh conversation.

**Resume Session**

Previous sessions are listed in the sidebar. Click to resume.

**Session History**

Sessions are persisted on the ChatRoom server and can be resumed even after browser refresh.

File Handling
-------------

**Viewing Files**

When agents reference files, they may be displayed inline or as links.

**File Uploads**

(Feature availability depends on toolset configuration)

Drag and drop files to upload them to the workspace.

Multi-Agent Display
-------------------

In team mode, the UI shows:

- Which agent is currently active
- Agent switching during delegation
- Clear attribution for each response

Settings
--------

**Theme**

Toggle between light and dark themes.

**Connection Status**

Indicator shows connection state to the ChatRoom server.

Keyboard Shortcuts
------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Action
   * - ``Enter``
     - Send message
   * - ``Shift+Enter``
     - New line in message
   * - ``Esc``
     - Clear input

Tips
----

**Long Conversations**

For long conversations, the agent may need to compress context. This happens automatically but can also be triggered from the REPL.

**Multiple Tabs**

You can open multiple tabs connected to the same ChatRoom for different conversations.

**Browser Refresh**

Refreshing the page maintains your connection. Just re-enter the service ID if disconnected.
