#![warn(clippy::missing_docs_in_private_items)]

//! This is an alternative implementation of code implemented in the blog post and source code
//! found at https://github.com/jamesmunns/trait-machine.

/// This constant were defined in the `impl` blocks of the types in the original source,
/// and has been moved up here for aesthetics more than anything.
/// https://github.com/jamesmunns/trait-machine
const SECTOR_SIZE: usize = 15;

/// This constant were defined in the `impl` blocks of the types in the original source,
/// and has been moved up here for aesthetics more than anything.
/// https://github.com/jamesmunns/trait-machine
const CHUNK_SIZE: usize = 5;

/// This type was taken directly from the original source:
/// https://github.com/jamesmunns/trait-machine
#[derive(Debug, PartialEq)]
pub enum Host2Client {
  #[allow(clippy::missing_docs_in_private_items)]
  Start { total_size: usize },
  #[allow(clippy::missing_docs_in_private_items)]
  EraseSector { addr: usize, len: usize },
  #[allow(clippy::missing_docs_in_private_items)]
  WriteData { addr: usize, data: Vec<u8> },
  #[allow(clippy::missing_docs_in_private_items)]
  Boot,
  #[allow(clippy::missing_docs_in_private_items)]
  Abort,
}

/// This type was taken directly from the original source:
/// https://github.com/jamesmunns/trait-machine
#[derive(Debug)]
pub enum Client2Host {
  #[allow(clippy::missing_docs_in_private_items)]
  ErrorReset,
  #[allow(clippy::missing_docs_in_private_items)]
  Starting,
  #[allow(clippy::missing_docs_in_private_items)]
  ChunkWritten,
  #[allow(clippy::missing_docs_in_private_items)]
  SectorErased,
  #[allow(clippy::missing_docs_in_private_items)]
  Booting,
}

/// This module defines the abstract, generic state machine.
mod eff {
  /// A generic definition of state machines. Note that the associated types `Message` and
  /// `Command` should ideally be associated with one another: `Commands` would be generic over the
  /// messages they produce, where that type parameter is "filled in" by this trait.
  ///
  /// I have (@dadleyy) tried exploring this in my "costanza" application that can be found here:
  /// https://github.com/dadleyy/costanza/blob/main/src/costanza-mid/src/eff.rs
  ///
  /// It is also worth nothing that the `iced` project has done a really good job so far with these
  /// concepts too:
  /// https://github.com/iced-rs/iced
  pub trait StateMachine {
    /// The kinds of things that our state machine needs to be initialized with.
    type Flags;
    /// The kinds of things that our state machine will react to.
    type Message;
    /// The kinds of things that our state machine will want to do.
    type Command;

    /// This function allows the creation of a state machine, with an optional starting effect.
    fn init(f: Self::Flags) -> (Self, Option<Self::Command>)
    where
      Self: Sized;

    /// This is the main function for our state machine.
    fn update(self, m: Self::Message) -> (Self, Option<Self::Command>)
    where
      Self: Sized;
  }
}

/// This is the "state machine" as actually expressed in terms of an_enumerated_ type.
#[derive(Debug, PartialEq)]
pub enum HostState {
  /// This is our initial state, returned by `init`.
  Empty {
    /// The buffer our state machine is working on.
    buffer: Vec<u8>,
  },
  /// The state we will belong to after we have requested a sector be erased.
  ErasingSector {
    /// The buffer our state machine is working on.
    buffer: Vec<u8>,
    /// The last position we erased.
    position: usize,
  },
  /// The most "meaty" state, where we are writing chunks one by one until we reach the end of our
  /// current sector.
  WritingChunks {
    /// The buffer our state machine is working on.
    buffer: Vec<u8>,
    /// The last position we erased.
    sector_offset: usize,
    /// The last chunk position we wrote.
    chunk_position: usize,
  },
  /// A state we go to when we have finished.
  Done,
  /// A state we go to when anything other than what we expect to happen, happens. In a real
  /// implemention, it is likely that we would try gracefully returning to `Empty` from the other
  /// states, which would re-sent our `Start` message.
  Error,
}

impl eff::StateMachine for HostState {
  type Flags = Vec<u8>;
  type Message = Client2Host;
  type Command = Host2Client;

  /// Prepares the state machine to a "default" state.
  fn init(data: Self::Flags) -> (Self, Option<Self::Command>) {
    let total_size = data.len();
    (Self::Empty { buffer: data }, Some(Host2Client::Start { total_size }))
  }

  /// The main function.
  fn update(self, message: Self::Message) -> (Self, Option<Self::Command>) {
    match (self, message) {
      // After `init` command is processed, we're empty and waiting for that "starting"
      // confirmation.
      (Self::Empty { buffer }, Client2Host::Starting) => {
        let cmd = Some(Host2Client::EraseSector {
          addr: 0,
          len: SECTOR_SIZE,
        });
        (Self::ErasingSector { buffer, position: 0 }, cmd)
      }

      // When we're erasing sectors and we receive the confirmation it has been erased,
      // start moving into chunk writing.
      (Self::ErasingSector { buffer, position }, Client2Host::SectorErased) => {
        let slice = &buffer[position..][..CHUNK_SIZE];
        let data = slice.to_vec();
        let cmd = Some(Host2Client::WriteData { addr: position, data });

        (
          Self::WritingChunks {
            buffer,
            sector_offset: position,
            chunk_position: CHUNK_SIZE,
          },
          cmd,
        )
      }

      // Here we are in the process of writing chunks and have received confirmation that it was
      // written successfully, attempt to do the next or move onto the next sector.
      (
        Self::WritingChunks {
          buffer,
          sector_offset,
          chunk_position,
        },
        Client2Host::ChunkWritten,
      ) => {
        let start = sector_offset + chunk_position;

        // If we're still within our sector, get the next buffer, and send that as the `WriteData`
        if start < sector_offset + SECTOR_SIZE {
          let slice = &buffer[start..][..CHUNK_SIZE];
          let data = slice.to_vec();
          let cmd = Some(Host2Client::WriteData { addr: start, data });
          return (
            Self::WritingChunks {
              buffer,
              sector_offset,
              chunk_position: chunk_position + CHUNK_SIZE,
            },
            cmd,
          );
        }

        // Otherwise, we need to check if we're completely done.
        if sector_offset + chunk_position >= buffer.len() {
          return (Self::Done, None);
        }

        let new_sector_start = sector_offset + SECTOR_SIZE;
        // Otherwise, we should start erasing the next sector.
        let cmd = Some(Host2Client::EraseSector {
          addr: new_sector_start,
          len: SECTOR_SIZE,
        });
        (
          Self::ErasingSector {
            buffer,
            position: new_sector_start,
          },
          cmd,
        )
      }

      (_, _) => (Self::Error, None),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::eff::StateMachine;
  use super::{Client2Host, Host2Client, HostState, CHUNK_SIZE, SECTOR_SIZE};

  #[test]
  fn test_happy() {
    let sector_count = 2;
    let mut buffer = Vec::with_capacity(SECTOR_SIZE * sector_count);
    let mut i = 0;
    buffer.resize_with(SECTOR_SIZE * sector_count, || {
      i += 1;
      i
    });

    let (mut host, mut command) = HostState::init(buffer.clone());

    assert_eq!(
      command,
      Some(Host2Client::Start {
        total_size: SECTOR_SIZE * sector_count
      })
    );
    assert!(matches!(host, HostState::Empty { .. }));

    (host, command) = host.update(Client2Host::Starting);
    assert!(
      matches!(host, HostState::ErasingSector { buffer: _, position: 0 }),
      "we're erasing a sector"
    );
    assert!(
      matches!(
        command,
        Some(Host2Client::EraseSector {
          addr: 0,
          len: SECTOR_SIZE
        })
      ),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::SectorErased);

    assert_eq!(
      host,
      HostState::WritingChunks {
        buffer: buffer.clone(),
        sector_offset: 0,
        chunk_position: CHUNK_SIZE
      },
      "we've moved into writing chunks"
    );
    assert_eq!(
      command,
      Some(Host2Client::WriteData {
        addr: 0,
        data: vec![1, 2, 3, 4, 5],
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::ChunkWritten);

    assert_eq!(
      host,
      HostState::WritingChunks {
        buffer: buffer.clone(),
        sector_offset: 0,
        chunk_position: CHUNK_SIZE + CHUNK_SIZE,
      },
      "weve started writing the SECOND chunk"
    );
    assert_eq!(
      command,
      Some(Host2Client::WriteData {
        addr: CHUNK_SIZE,
        data: vec![6, 7, 8, 9, 10],
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::ChunkWritten);

    assert_eq!(
      host,
      HostState::WritingChunks {
        buffer: buffer.clone(),
        sector_offset: 0,
        chunk_position: CHUNK_SIZE + CHUNK_SIZE + CHUNK_SIZE,
      },
      "weve started writing the THIRD chunk"
    );
    assert_eq!(
      command,
      Some(Host2Client::WriteData {
        addr: CHUNK_SIZE + CHUNK_SIZE,
        data: vec![11, 12, 13, 14, 15],
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::ChunkWritten);

    assert_eq!(
      host,
      HostState::ErasingSector {
        buffer: buffer.clone(),
        position: SECTOR_SIZE,
      },
      "weve started erasing the second chunk"
    );
    assert_eq!(
      command,
      Some(Host2Client::EraseSector {
        addr: SECTOR_SIZE,
        len: SECTOR_SIZE,
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::SectorErased);

    assert_eq!(
      host,
      HostState::WritingChunks {
        buffer: buffer.clone(),
        sector_offset: SECTOR_SIZE,
        chunk_position: CHUNK_SIZE,
      },
      "weve started writing the FIRST chunk of SECOND sector"
    );
    assert_eq!(
      command,
      Some(Host2Client::WriteData {
        addr: SECTOR_SIZE,
        data: vec![16, 17, 18, 19, 20],
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::ChunkWritten);

    assert_eq!(
      host,
      HostState::WritingChunks {
        buffer: buffer.clone(),
        sector_offset: SECTOR_SIZE,
        chunk_position: CHUNK_SIZE + CHUNK_SIZE,
      },
      "weve started writing the SECOND chunk of SECOND sector"
    );
    assert_eq!(
      command,
      Some(Host2Client::WriteData {
        addr: SECTOR_SIZE + CHUNK_SIZE,
        data: vec![21, 22, 23, 24, 25],
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::ChunkWritten);

    assert_eq!(
      host,
      HostState::WritingChunks {
        buffer: buffer.clone(),
        sector_offset: SECTOR_SIZE,
        chunk_position: CHUNK_SIZE + CHUNK_SIZE + CHUNK_SIZE,
      },
      "weve started writing the THIRD chunk of SECOND sector"
    );
    assert_eq!(
      command,
      Some(Host2Client::WriteData {
        addr: SECTOR_SIZE + CHUNK_SIZE + CHUNK_SIZE,
        data: vec![26, 27, 28, 29, 30],
      }),
      "the correct command was sent"
    );

    (host, command) = host.update(Client2Host::ChunkWritten);

    assert_eq!(host, HostState::Done, "we are now done");
    assert_eq!(command, None, "the correct command was sent");
  }
}
